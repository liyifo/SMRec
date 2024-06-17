import torch
import numpy as np
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
from models.DSNTSP.temporal_set_prediction import temporal_set_prediction
from load_data import *
from opt_einsum import contract


class GraphConvolution(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class GCN(torch.nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)

        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = torch.nn.Linear(input_dim, embed_dim)
        self.key = torch.nn.Linear(input_dim, embed_dim)
        self.value = torch.nn.Linear(input_dim, embed_dim)
        self.model_dim = embed_dim

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_scores = torch.matmul(q, k.transpose(0,1)) / math.sqrt(self.model_dim)
        attn_scores = F.softmax(attn_scores, dim=-1)

        aggregated = torch.matmul(attn_scores, v)
        return aggregated

class MainModel(torch.nn.Module):

    def __init__(self, device ,voc_size, emb_dim, ehr_adj, ddi_adj, dropout=0.5):
        '''
        gpu, 词表大小，嵌入维度, 嵌入dropout 
        '''
        super(MainModel, self).__init__()
        self.device = device
        self.med_size = voc_size[2]
        self.emb_dim = emb_dim
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # dp embedding module
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim)
        ])
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        self.dropout =  torch.nn.Dropout(p=dropout) if dropout > 0 and dropout < 1 else torch.nn.Sequential()
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 4, emb_dim)
        )
        self.vlevel_lenear = torch.nn.Linear(emb_dim*2, 1)

        # ehr ddi encode module
        self.gcn =  GCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        self.inter = torch.nn.Parameter(torch.FloatTensor(1))
        self.inter2 = torch.nn.Parameter(torch.FloatTensor(1))
        # self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)


        # medicine encode module
        self.med_embedding = torch.nn.Embedding(self.med_size, emb_dim)
        self.tsp = temporal_set_prediction(self.med_size, emb_dim)

        self.ln = torch.nn.LayerNorm(self.med_size)


    def forward(self, patient_data):
        # 输入的是multi-label向量
        # 每次输入一个病人的数据，对每一次visit进行嵌入
        d_seq, p_seq = [], []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device) # (1,d_len)
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)
            repr1 = self.dropout(self.embeddings[0](Idx1)) # (1, d_len, emb_size)
            repr2 = self.dropout(self.embeddings[1](Idx2))
            # 对每个d和q的嵌入进行加和
            d_seq.append(torch.sum(repr1, keepdim=True, dim=1)) # (visit_num, 1, 1, emb_size)
            p_seq.append(torch.sum(repr2, keepdim=True, dim=1))
        d_seq = torch.cat(d_seq, dim=1) # （1, visit_num, emb_size)
        p_seq = torch.cat(p_seq, dim=1)
        
        output1, hidden1 = self.seq_encoders[0](d_seq) # (1, visit_num, emb_size)
        output2, hidden2 = self.seq_encoders[1](p_seq)
        # print('***output1***', output1.shape)
        # print('***hidden1***', hidden1.shape)
        # print('***output2***', output2.shape)
        # print('***hidden2***', hidden2.shape)
        # 拼接dp表征
        seq_repr = torch.cat([hidden1, hidden2], dim=-1) # (1, 1, emb_size*2)
        # 最后一行的输出作为最后的表征
        last_repr = torch.cat([output1[:, -1],  output2[:, -1]], dim=-1) # (1, emb_size*2)
        # 生成病人表征，即隐藏层和最终输出的拼接
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()]) # (emb_size*4)
        query = self.query(patient_repr) # (emb_size)

        # visit level  应该放入隐藏层
        # visit_level = self.vlevel_lenear(torch.cat([output1, output2], dim=1).unsqueeze(0)) #(visit_num, 1)
        # print('***visit_level***', visit_level.shape)

        ehr_embedding, ddi_embedding = self.gcn()
        drug_memory = ehr_embedding - ddi_embedding * self.inter
        # drug_memory_padding = torch.zeros((3, self.emb_dim), device=self.device).float()
        # drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)

        med = []
        for i in range(len(patient_data)):
            med.append(patient_data[i][2])
        g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = collate_set_across_user([dnntsp_process(med, self.med_embedding, self.device)], self.med_size, self.device)
        medicine_repr = self.tsp(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)

        # cat
        inter2 = torch.sigmoid(self.inter2)
        medicine_repr = medicine_repr*inter2 + drug_memory*(1-inter2)


        # prediction
        #  (1,64) * (131,64) = (1,131)

        result = torch.matmul(query.unsqueeze(0), medicine_repr.t())

        result = self.ln(result)
        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return result, batch_neg
        


class MainModel_attn_only(torch.nn.Module):

    def __init__(self, device ,voc_size, emb_dim, ehr_adj, ddi_adj, dropout=0.5):
        '''
        gpu, 词表大小，嵌入维度, 嵌入dropout， 
        '''
        super(MainModel_attn_only, self).__init__()
        self.device = device
        self.med_size = voc_size[2]
        self.emb_dim = emb_dim
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # dp embedding module
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim)
        ])
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        self.dropout =  torch.nn.Dropout(p=dropout) if dropout > 0 and dropout < 1 else torch.nn.Sequential()
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 4, emb_dim)
        )

        # ehr ddi encode module
        self.gcn =  GCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        # self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)


        # medicine encode module
        self.med_embedding = torch.nn.Embedding(self.med_size, emb_dim)
        self.tsp = temporal_set_prediction(self.med_size, emb_dim)
        self.attn = SelfAttention(emb_dim*3, emb_dim)
        self.w_linear = torch.nn.Linear(emb_dim, emb_dim)
        self.b_linear = torch.nn.Linear(emb_dim, 1)

        self.ln = torch.nn.LayerNorm(self.med_size)


    def forward(self, patient_data):
        # 输入的是multi-label向量
        # 每次输入一个病人的数据，对每一次visit进行嵌入
        d_seq, p_seq = [], []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device) # (1,d_len)
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)
            repr1 = self.dropout(self.embeddings[0](Idx1)) # (1, d_len, emb_size)
            repr2 = self.dropout(self.embeddings[1](Idx2))
            # 对每个d和q的嵌入进行加和
            d_seq.append(torch.sum(repr1, keepdim=True, dim=1)) # (visit_num, 1, 1, emb_size)
            p_seq.append(torch.sum(repr2, keepdim=True, dim=1))
        d_seq = torch.cat(d_seq, dim=1) # （1, visit_num, emb_size)
        p_seq = torch.cat(p_seq, dim=1)
        
        output1, hidden1 = self.seq_encoders[0](d_seq) # (1, visit_num, emb_size)
        output2, hidden2 = self.seq_encoders[1](p_seq)

        # 拼接dp表征
        seq_repr = torch.cat([hidden1, hidden2], dim=-1) # (1, 1, emb_size*2)
        # 最后一行的输出作为最后的表征
        last_repr = torch.cat([output1[:, -1],  output2[:, -1]], dim=-1) # (1, emb_size*2)
        # 生成病人表征，即隐藏层和最终输出的拼接
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()]) # (emb_size*4)
        query = self.query(patient_repr).unsqueeze(0) # (1, emb_size)


        ehr_embedding, ddi_embedding = self.gcn()


        med = []
        for i in range(len(patient_data)):
            med.append(patient_data[i][2])
        g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = collate_set_across_user([dnntsp_process(med, self.med_embedding, self.device)], self.med_size, self.device)
        medicine_repr = self.tsp(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)

        # attn
        medicine_repr = torch.cat([ehr_embedding, ddi_embedding, medicine_repr], dim=-1) # (131, 64*3)
        medicine_repr = self.attn(medicine_repr) # (131,64)
        # print(medicine_repr.shape)


        result = torch.matmul(query, medicine_repr.t())
        result = self.ln(result)
        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return result, batch_neg
        




class MainModel_attn(torch.nn.Module):
    

    def __init__(self, device ,voc_size, emb_dim, ehr_adj, ddi_adj, dropout=0.5):
        '''
        gpu, 词表大小，嵌入维度, 嵌入dropout， 
        '''
        super(MainModel_attn, self).__init__()
        self.device = device
        self.med_size = voc_size[2]
        self.emb_dim = emb_dim
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # dp embedding module
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim)
        ])
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        self.dropout =  torch.nn.Dropout(p=dropout) if dropout > 0 and dropout < 1 else torch.nn.Sequential()
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 4, emb_dim)
        )

        # ehr ddi encode module
        self.gcn =  GCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        # self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)


        # medicine encode module
        self.med_embedding = torch.nn.Embedding(self.med_size, emb_dim)
        self.tsp = temporal_set_prediction(self.med_size, emb_dim)
        self.attn = SelfAttention(emb_dim*3, emb_dim)
        self.w_linear = torch.nn.Linear(emb_dim, emb_dim)
        self.b_linear = torch.nn.Linear(emb_dim, 1)

        self.ln = torch.nn.LayerNorm(self.med_size)


    def forward(self, patient_data):
        # 输入的是multi-label向量
        # 每次输入一个病人的数据，对每一次visit进行嵌入
        d_seq, p_seq = [], []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device) # (1,d_len)
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)
            repr1 = self.dropout(self.embeddings[0](Idx1)) # (1, d_len, emb_size)
            repr2 = self.dropout(self.embeddings[1](Idx2))
            # 对每个d和q的嵌入进行加和
            d_seq.append(torch.sum(repr1, keepdim=True, dim=1)) # (visit_num, 1, 1, emb_size)
            p_seq.append(torch.sum(repr2, keepdim=True, dim=1))
        d_seq = torch.cat(d_seq, dim=1) # （1, visit_num, emb_size)
        p_seq = torch.cat(p_seq, dim=1)
        
        output1, hidden1 = self.seq_encoders[0](d_seq) # (1, visit_num, emb_size)
        output2, hidden2 = self.seq_encoders[1](p_seq)

        # 拼接dp表征
        seq_repr = torch.cat([hidden1, hidden2], dim=-1) # (1, 1, emb_size*2)
        # 最后一行的输出作为最后的表征
        last_repr = torch.cat([output1[:, -1],  output2[:, -1]], dim=-1) # (1, emb_size*2)
        # 生成病人表征，即隐藏层和最终输出的拼接
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()]) # (emb_size*4)
        query = self.query(patient_repr).unsqueeze(0) # (1, emb_size)


        ehr_embedding, ddi_embedding = self.gcn()


        med = []
        for i in range(len(patient_data)):
            med.append(patient_data[i][2])
        g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = collate_set_across_user([dnntsp_process(med, self.med_embedding, self.device)], self.med_size, self.device)
        medicine_repr = self.tsp(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)

        # attn
        medicine_repr = torch.cat([ehr_embedding, ddi_embedding, medicine_repr], dim=-1) # (131, 64*3)
        medicine_repr = self.attn(medicine_repr) # (131,64)
        # print(medicine_repr.shape)


        # prediction
        #  

        query = query.unsqueeze(0) # 1 1 64
        score = contract('abc, dc->adc', query, medicine_repr)  # 1 131 64
        score = torch.softmax(score, dim=-1)
        m = contract('abc, adc->adc', query, score)  # 1 131 64

        w = self.w_linear(medicine_repr)
        b = self.b_linear(medicine_repr)
        result = self.get_logits(m, w, b) # 1 131


        # result = self.ln(result)
        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return result, batch_neg

    def get_logits(self, m, w, b):
        return contract('blh, lh->bl', m, w) + b.squeeze(-1)
    


class MainModel_level(torch.nn.Module):

    def __init__(self, device ,voc_size, emb_dim, ehr_adj, ddi_adj, dropout=0.5):
        '''
        gpu, 词表大小，嵌入维度, 嵌入dropout， 
        '''
        super(MainModel_level, self).__init__()
        self.device = device
        self.med_size = voc_size[2]
        self.emb_dim = emb_dim
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # dp embedding module
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim)
        ])
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        self.dropout =  torch.nn.Dropout(p=dropout) if dropout > 0 and dropout < 1 else torch.nn.Sequential()
        self.query = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim)
        )
        self.vlevel_linear = torch.nn.Linear(emb_dim*2, 1)
        self.mlevel_linear = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_dim * 2, voc_size[2])
        )

        # ehr ddi encode module
        self.gcn =  GCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        self.inter = torch.nn.Parameter(torch.FloatTensor(1))
        self.inter2 = torch.nn.Parameter(torch.FloatTensor(1))
        # self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)


        # medicine encode module
        self.med_embedding = torch.nn.Embedding(self.med_size, emb_dim)
        self.tsp = temporal_set_prediction(self.med_size, emb_dim)

        self.ln = torch.nn.LayerNorm(self.med_size)


    def forward(self, patient_data):
        # 输入的是multi-label向量
        # 每次输入一个病人的数据，对每一次visit进行嵌入
        d_seq, p_seq = [], []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device) # (1,d_len)
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)
            repr1 = self.dropout(self.embeddings[0](Idx1)) # (1, d_len, emb_size)
            repr2 = self.dropout(self.embeddings[1](Idx2))
            # 对每个d和q的嵌入进行加和
            d_seq.append(torch.sum(repr1, keepdim=True, dim=1)) # (visit_num, 1, 1, emb_size)
            p_seq.append(torch.sum(repr2, keepdim=True, dim=1))
        d_seq = torch.cat(d_seq, dim=1) # （1, visit_num, emb_size)
        p_seq = torch.cat(p_seq, dim=1)
        
        output1, hidden1 = self.seq_encoders[0](d_seq) # (1, visit_num, emb_size)
        output2, hidden2 = self.seq_encoders[1](p_seq)


        # 拼接dp表征
        seq_repr = torch.cat([hidden1, hidden2], dim=-1) # (1, 1, emb_size*2)
        query = self.query(seq_repr.flatten()).unsqueeze(0) # (1, emb_size)

        seq_repr = torch.cat([output1[:,:-1,:], output2[:,:-1,:]], dim=-1).squeeze(0) # (1, visit_num, emb_size*2)
        med_query = self.query(seq_repr) # (visit_num, emb_size)
        # med_level = torch.cat([output1, output2], dim=-1) # (1,visit_num,emb_size*2)
        # med_level = self.mlevel_linear(med_level) # (1,visit_num,131)
        # visit level  应该放入隐藏层
        
        


        ehr_embedding, ddi_embedding = self.gcn()
        drug_memory = ehr_embedding - ddi_embedding * self.inter
        # drug_memory_padding = torch.zeros((3, self.emb_dim), device=self.device).float()
        # drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)

        med = []
        for i in range(len(patient_data)):
            med.append(patient_data[i][2])
        g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = collate_set_across_user([dnntsp_process(med, self.med_embedding, self.device)], self.med_size, self.device)
        
        med_emb = self.med_embedding(nodes.clone()).to(self.device) # (med_size, emb_size)
        med_level = torch.matmul(med_emb, med_query.t())
        # print(query.shape)
        
        medicine_repr = self.tsp(g, nodes_feature, edges_weight, lengths, nodes, users_frequency, med_level)

        # cat
        inter2 = torch.sigmoid(self.inter2)
        medicine_repr = medicine_repr*inter2 + drug_memory*(1-inter2)
        # print(self.inter)

        # prediction
        #  (1,64) * (131,64) = (1,131)

        result = torch.matmul(query, medicine_repr.t())

        result = self.ln(result)
        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return result, batch_neg
        

class MainModel_level_no_share(torch.nn.Module):

    def __init__(self, device ,voc_size, emb_dim, ehr_adj, ddi_adj, dropout=0.5):
        '''
        gpu, 词表大小，嵌入维度, 嵌入dropout 
        '''
        super(MainModel_level_no_share, self).__init__()
        self.device = device
        self.med_size = voc_size[2]
        self.emb_dim = emb_dim
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # dp embedding module
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim)
        ])
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        self.dropout =  torch.nn.Dropout(p=dropout) if dropout > 0 and dropout < 1 else torch.nn.Sequential()
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim)
        )
        self.query2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim)
        )
        self.vlevel_linear = torch.nn.Linear(emb_dim*2, 1)
        self.mlevel_linear = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, voc_size[2])
        )

        # ehr ddi encode module
        self.gcn =  GCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        self.inter = torch.nn.Parameter(torch.FloatTensor(1))
        self.inter2 = torch.nn.Parameter(torch.FloatTensor(1))
        # self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

        
        # medicine encode module
        self.med_embedding = torch.nn.Embedding(self.med_size, emb_dim)
        self.tsp = temporal_set_prediction(self.med_size, emb_dim)
        self.ln = torch.nn.LayerNorm(self.med_size)
        self.init_weights()


    def forward(self, patient_data):
        # 输入的是multi-label向量
        # 每次输入一个病人的数据，对每一次visit进行嵌入
        d_seq, p_seq = [], []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device) # (1,d_len)
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)
            repr1 = self.dropout(self.embeddings[0](Idx1)) # (1, d_len, emb_size)
            repr2 = self.dropout(self.embeddings[1](Idx2))
            # 对每个d和q的嵌入进行加和
            d_seq.append(torch.sum(repr1, keepdim=True, dim=1)) # (visit_num, 1, 1, emb_size)
            p_seq.append(torch.sum(repr2, keepdim=True, dim=1))
        d_seq = torch.cat(d_seq, dim=1) # （1, visit_num, emb_size)
        p_seq = torch.cat(p_seq, dim=1)
        
        output1, hidden1 = self.seq_encoders[0](d_seq) # (1, visit_num, emb_size)
        output2, hidden2 = self.seq_encoders[1](p_seq)


        # 拼接dp表征
        seq_repr = torch.cat([hidden1, hidden2], dim=-1) # (1, 1, emb_size*2)
        query = self.query(seq_repr.flatten()).unsqueeze(0) # (1, emb_size)

        seq_repr = torch.cat([output1[:,:-1,:], output2[:,:-1,:]], dim=-1).squeeze(0) # (1, visit_num, emb_size*2)
        med_query = self.query2(seq_repr) # (visit_num, emb_size)
        # med_level = torch.cat([output1, output2], dim=-1) # (1,visit_num,emb_size*2)
        # med_level = self.mlevel_linear(med_level) # (1,visit_num,131)
        # visit level  应该放入隐藏层
        
        


        ehr_embedding, ddi_embedding = self.gcn()
        drug_memory = ehr_embedding - ddi_embedding * self.inter
        # drug_memory = ehr_embedding*(1-self.inter) + ddi_embedding * self.inter
        # drug_memory_padding = torch.zeros((3, self.emb_dim), device=self.device).float()
        # drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)

        med = []
        for i in range(len(patient_data)):
            med.append(patient_data[i][2])
        g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = collate_set_across_user([dnntsp_process(med, self.med_embedding, self.device)], self.med_size, self.device)
        
        med_emb = self.med_embedding(nodes.clone()).to(self.device) # (med_size, emb_size)
        med_level = torch.matmul(med_emb, med_query.t())
        # print(query.shape)
        
        medicine_repr = self.tsp(g, nodes_feature, edges_weight, lengths, nodes, users_frequency, med_level)

        # cat
        inter2 = torch.sigmoid(self.inter2)
        medicine_repr = medicine_repr*inter2 + drug_memory*(1-inter2)
        #medicine_repr = medicine_repr + drug_memory*self.inter2
        # print(self.inter)

        # prediction
        #  (1,64) * (131,64) = (1,131)

        result = torch.matmul(query, medicine_repr.t())

        result = self.ln(result)
        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return result, batch_neg
    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        



class MainModel_level_no_share_new(torch.nn.Module):

    def __init__(self, device ,voc_size, emb_dim, ehr_adj, ddi_adj, dropout=0.5):
        '''
        gpu, 词表大小，嵌入维度, 嵌入dropout， 
        '''
        super(MainModel_level_no_share_new, self).__init__()
        self.device = device
        self.med_size = voc_size[2]
        self.emb_dim = emb_dim
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # dp embedding module
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim)
        ])
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        self.dropout =  torch.nn.Dropout(p=dropout) if dropout > 0 and dropout < 1 else torch.nn.Sequential()
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim)
        )
        self.query2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim)
        )
        self.vlevel_linear = torch.nn.Linear(emb_dim*2, 1)
        self.mlevel_linear = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, voc_size[2])
        )

        # ehr ddi encode module
        self.gcn =  GCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        self.inter = torch.nn.Parameter(torch.FloatTensor(1))
        self.inter2 = torch.nn.Parameter(torch.FloatTensor(1))
        # self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

        
        # medicine encode module
        # self.med_embedding = torch.nn.Embedding(self.med_size, emb_dim)
        self.tsp = temporal_set_prediction(self.med_size, emb_dim)
        self.ln = torch.nn.LayerNorm(self.med_size)
        self.init_weights()


    def forward(self, patient_data):
        # 输入的是multi-label向量
        # 每次输入一个病人的数据，对每一次visit进行嵌入
        d_seq, p_seq = [], []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device) # (1,d_len)
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)
            repr1 = self.dropout(self.embeddings[0](Idx1)) # (1, d_len, emb_size)
            repr2 = self.dropout(self.embeddings[1](Idx2))
            # 对每个d和q的嵌入进行加和
            d_seq.append(torch.sum(repr1, keepdim=True, dim=1)) # (visit_num, 1, 1, emb_size)
            p_seq.append(torch.sum(repr2, keepdim=True, dim=1))
        d_seq = torch.cat(d_seq, dim=1) # （1, visit_num, emb_size)
        p_seq = torch.cat(p_seq, dim=1)
        
        output1, hidden1 = self.seq_encoders[0](d_seq) # (1, visit_num, emb_size)
        output2, hidden2 = self.seq_encoders[1](p_seq)


        # 拼接dp表征
        seq_repr = torch.cat([hidden1, hidden2], dim=-1) # (1, 1, emb_size*2)
        query = self.query(seq_repr.flatten()).unsqueeze(0) # (1, emb_size)

        seq_repr = torch.cat([output1[:,:-1,:], output2[:,:-1,:]], dim=-1).squeeze(0) # (1, visit_num, emb_size*2)
        med_query = self.query2(seq_repr) # (visit_num, emb_size)
        # med_level = torch.cat([output1, output2], dim=-1) # (1,visit_num,emb_size*2)
        # med_level = self.mlevel_linear(med_level) # (1,visit_num,131)
        # visit level  应该放入隐藏层
        
        


        ehr_embedding, ddi_embedding = self.gcn()
        drug_memory = ehr_embedding - ddi_embedding * self.inter
        # drug_memory = ehr_embedding*(1-self.inter) + ddi_embedding * self.inter
        # drug_memory_padding = torch.zeros((3, self.emb_dim), device=self.device).float()
        # drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)

        med = []
        for i in range(len(patient_data)):
            med.append(patient_data[i][2])
        g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = collate_set_across_user([dnntsp_process(med, self.tsp.item_embedding, self.device)], self.med_size, self.device)
        
        med_emb = self.tsp.item_embedding(nodes.clone()).to(self.device) # (med_size, emb_size)
        med_level = torch.matmul(med_emb, med_query.t())
        # print(query.shape)
        
        medicine_repr = self.tsp(g, nodes_feature, edges_weight, lengths, nodes, users_frequency, med_level)

        # cat
        inter2 = torch.sigmoid(self.inter2)
        medicine_repr = medicine_repr*inter2 + drug_memory*(1-inter2)
        #medicine_repr = medicine_repr + drug_memory*self.inter2
        # print(self.inter)

        # prediction
        #  (1,64) * (131,64) = (1,131)

        result = torch.matmul(query, medicine_repr.t())

        result = self.ln(result)
        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return result, batch_neg
    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)



class abalation_no_global(torch.nn.Module):

    def __init__(self, device ,voc_size, emb_dim, ehr_adj, ddi_adj, dropout=0.5):
        '''
        gpu, 词表大小，嵌入维度, 嵌入dropout， 
        '''
        super(MainModel_level_no_share_new, self).__init__()
        self.device = device
        self.med_size = voc_size[2]
        self.emb_dim = emb_dim
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # dp embedding module
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim)
        ])
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        self.dropout =  torch.nn.Dropout(p=dropout) if dropout > 0 and dropout < 1 else torch.nn.Sequential()
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim)
        )
        self.query2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim)
        )
        self.vlevel_linear = torch.nn.Linear(emb_dim*2, 1)
        self.mlevel_linear = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, voc_size[2])
        )

        # ehr ddi encode module
        self.gcn =  GCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        self.inter = torch.nn.Parameter(torch.FloatTensor(1))
        self.inter2 = torch.nn.Parameter(torch.FloatTensor(1))
        # self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

        
        # medicine encode module
        # self.med_embedding = torch.nn.Embedding(self.med_size, emb_dim)
        self.tsp = temporal_set_prediction(self.med_size, emb_dim)
        self.ln = torch.nn.LayerNorm(self.med_size)
        self.init_weights()


    def forward(self, patient_data):
        # 输入的是multi-label向量
        # 每次输入一个病人的数据，对每一次visit进行嵌入
        d_seq, p_seq = [], []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device) # (1,d_len)
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)
            repr1 = self.dropout(self.embeddings[0](Idx1)) # (1, d_len, emb_size)
            repr2 = self.dropout(self.embeddings[1](Idx2))
            # 对每个d和q的嵌入进行加和
            d_seq.append(torch.sum(repr1, keepdim=True, dim=1)) # (visit_num, 1, 1, emb_size)
            p_seq.append(torch.sum(repr2, keepdim=True, dim=1))
        d_seq = torch.cat(d_seq, dim=1) # （1, visit_num, emb_size)
        p_seq = torch.cat(p_seq, dim=1)
        
        output1, hidden1 = self.seq_encoders[0](d_seq) # (1, visit_num, emb_size)
        output2, hidden2 = self.seq_encoders[1](p_seq)


        # 拼接dp表征
        seq_repr = torch.cat([hidden1, hidden2], dim=-1) # (1, 1, emb_size*2)
        query = self.query(seq_repr.flatten()).unsqueeze(0) # (1, emb_size)

        seq_repr = torch.cat([output1[:,:-1,:], output2[:,:-1,:]], dim=-1).squeeze(0) # (1, visit_num, emb_size*2)
        med_query = self.query2(seq_repr) # (visit_num, emb_size)
        # med_level = torch.cat([output1, output2], dim=-1) # (1,visit_num,emb_size*2)
        # med_level = self.mlevel_linear(med_level) # (1,visit_num,131)
        # visit level  应该放入隐藏层
        
        


        ehr_embedding, ddi_embedding = self.gcn()
        # drug_memory = ehr_embedding - ddi_embedding * self.inter
        drug_memory = ddi_embedding

        # drug_memory = ehr_embedding*(1-self.inter) + ddi_embedding * self.inter
        # drug_memory_padding = torch.zeros((3, self.emb_dim), device=self.device).float()
        # drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)

        med = []
        for i in range(len(patient_data)):
            med.append(patient_data[i][2])
        g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = collate_set_across_user([dnntsp_process(med, self.tsp.item_embedding, self.device)], self.med_size, self.device)
        
        med_emb = self.tsp.item_embedding(nodes.clone()).to(self.device) # (med_size, emb_size)
        med_level = torch.matmul(med_emb, med_query.t())
        # print(query.shape)
        
        medicine_repr = self.tsp(g, nodes_feature, edges_weight, lengths, nodes, users_frequency, med_level)

        # cat
        inter2 = torch.sigmoid(self.inter2)
        medicine_repr = medicine_repr*inter2 + drug_memory*(1-inter2)
        #medicine_repr = medicine_repr + drug_memory*self.inter2
        # print(self.inter)

        # prediction
        #  (1,64) * (131,64) = (1,131)

        result = torch.matmul(query, medicine_repr.t())

        result = self.ln(result)
        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return result, batch_neg
    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
