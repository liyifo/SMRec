
import torch
import dill
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
import itertools
from collections import defaultdict
import dgl
import numpy as np
import  torch.nn.functional as F
from sklearn.preprocessing import normalize

class MimicDataSet(data.Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
def mimic_collate(batch):
    seq_length = torch.tensor([len(data) for data in batch])
    batch_size = len(batch)
    max_seq = max(seq_length)

    # 统计每一个seq疾病、手术、药物的数量，以及相应的最值
    # 同时为每一个seq的disease计算与上一个seq的disease之间的交集和差集
    d_length_matrix = []
    p_length_matrix = []
    m_length_matrix = []
    d_max_num = 0
    p_max_num = 0
    m_max_num = 0
    d_dec_list = []
    d_stay_list = []
    p_dec_list = []
    p_stay_list = []
    for data in batch:
        # 对每个病人，统计所有visit的d p m的数量，并记录最大值
        d_buf, p_buf, m_buf = [], [], []
        d_dec_list_buf, d_stay_list_buf = [], []
        p_dec_list_buf, p_stay_list_buf = [], []
        for idx, seq in enumerate(data):
            d_buf.append(len(seq[0]))
            p_buf.append(len(seq[1]))
            m_buf.append(len(seq[2]))
            d_max_num = max(d_max_num, len(seq[0]))
            p_max_num = max(p_max_num, len(seq[1]))
            m_max_num = max(m_max_num, len(seq[2]))
            if idx==0:
                # 第一个seq，则交集与差集为空
                d_dec_list_buf.append([])
                d_stay_list_buf.append([])
                p_dec_list_buf.append([])
                p_stay_list_buf.append([])
            else:
                # 计算当前visit和上次visit差集与交集
                cur_d = set(seq[0])
                last_d = set(data[idx-1][0])
                stay_list = list(cur_d & last_d)
                dec_list = list(last_d - cur_d)
                d_dec_list_buf.append(dec_list)
                d_stay_list_buf.append(stay_list)

                cur_p = set(seq[1])
                last_p = set(data[idx-1][1])
                proc_stay_list = list(cur_p & last_p)
                proc_dec_list = list(last_p - cur_p)
                p_dec_list_buf.append(proc_dec_list)
                p_stay_list_buf.append(proc_stay_list)
        d_length_matrix.append(d_buf)
        p_length_matrix.append(p_buf)
        m_length_matrix.append(m_buf)
        d_dec_list.append(d_dec_list_buf)
        d_stay_list.append(d_stay_list_buf)
        p_dec_list.append(p_dec_list_buf)
        p_stay_list.append(p_stay_list_buf)

    # 生成m_mask_matrix
    m_mask_matrix = torch.full((batch_size, max_seq, m_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(m_length_matrix[i])):
            m_mask_matrix[i, j, :m_length_matrix[i][j]] = 0.

    # 生成d_mask_matrix
    d_mask_matrix = torch.full((batch_size, max_seq, d_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(d_length_matrix[i])):
            d_mask_matrix[i, j, :d_length_matrix[i][j]] = 0.

    # 生成p_mask_matrix
    p_mask_matrix = torch.full((batch_size, max_seq, p_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(p_length_matrix[i])):
            p_mask_matrix[i, j, :p_length_matrix[i][j]] = 0.

    # 分别生成dec_disease_tensor和stay_disease_tensor
    dec_disease_tensor = torch.full((batch_size, max_seq, d_max_num), -1)
    stay_disease_tensor = torch.full((batch_size, max_seq, d_max_num), -1)
    dec_disease_mask = torch.full((batch_size, max_seq, d_max_num), -1e9)
    stay_disease_mask = torch.full((batch_size, max_seq, d_max_num), -1e9)
    for b_id, (dec_seqs, stay_seqs) in enumerate(zip(d_dec_list, d_stay_list)):
        for s_id, (dec_adm, stay_adm) in enumerate(zip(dec_seqs, stay_seqs)):
            dec_disease_tensor[b_id, s_id, :len(dec_adm)] = torch.tensor(dec_adm)
            stay_disease_tensor[b_id, s_id, :len(stay_adm)] = torch.tensor(stay_adm)
            dec_disease_mask[b_id, s_id, :len(dec_adm)] = 0.
            stay_disease_mask[b_id, s_id, :len(dec_adm)] = 0.

    # 分别生成dec_disease_tensor和stay_disease_tensor
    dec_proc_tensor = torch.full((batch_size, max_seq, p_max_num), -1)
    stay_proc_tensor = torch.full((batch_size, max_seq, p_max_num), -1)
    dec_proc_mask = torch.full((batch_size, max_seq, p_max_num), -1e9)
    stay_proc_mask = torch.full((batch_size, max_seq, p_max_num), -1e9)
    for b_id, (dec_seqs, stay_seqs) in enumerate(zip(p_dec_list, p_stay_list)):
        for s_id, (dec_adm, stay_adm) in enumerate(zip(dec_seqs, stay_seqs)):
            dec_proc_tensor[b_id, s_id, :len(dec_adm)] = torch.tensor(dec_adm)
            stay_proc_tensor[b_id, s_id, :len(stay_adm)] = torch.tensor(stay_adm)
            dec_proc_mask[b_id, s_id, :len(dec_adm)] = 0.
            stay_proc_mask[b_id, s_id, :len(dec_adm)] = 0.

    # 分别生成disease、procedure、medication的数据
    disease_tensor = torch.full((batch_size, max_seq, d_max_num), -1)
    procedure_tensor = torch.full((batch_size, max_seq, p_max_num), -1)
    medication_tensor = torch.full((batch_size, max_seq, m_max_num), 0)

    # 分别拼接成一个batch的数据
    for b_id, data in enumerate(batch):
        for s_id, adm in enumerate(data):
            # adm部分的数据按照disease、procedure、medication排序
            disease_tensor[b_id, s_id, :len(adm[0])] = torch.tensor(adm[0])
            procedure_tensor[b_id, s_id, :len(adm[1])] = torch.tensor(adm[1])
            # dynamic shuffle
            # cur_medications = adm[2]
            # random.shuffle(cur_medications)
            # medication_tensor[b_id, s_id, :len(adm[2])] = torch.tensor(cur_medications)
            medication_tensor[b_id, s_id, :len(adm[2])] = torch.tensor(adm[2])

    # print(disease_tensor[1])
    # dpm元素矩阵（batch，访问，元素下标）   seq（batch，访问，dpm，元素）
    # d_length_matrix dpm长度矩阵 (batch，访问) = dqm长度
    # d_mask_matrix dpm掩码矩阵 (batch, 访问，max_len)
    # dp交差集矩阵 （batch，访问，交差集元素下标） 掩码矩阵(batch, 访问，max_len)
    return disease_tensor, procedure_tensor, medication_tensor, seq_length, \
        d_length_matrix, p_length_matrix, m_length_matrix, \
            d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                dec_disease_tensor, stay_disease_tensor, dec_disease_mask, stay_disease_mask, \
                    dec_proc_tensor, stay_proc_tensor, dec_proc_mask, stay_proc_mask

def load_mimic3(device, args):
    '''
    ddi矩阵，ddi mask矩阵， 
    '''
    data_path = "./data/MIMIC-III/records_final.pkl"
    voc_path = "./data//MIMIC-III/voc_final.pkl"

    ddi_adj_path = "./data/MIMIC-III/ddi_A_final.pkl"
    ddi_mask_path = "./data/MIMIC-III/ddi_mask_H.pkl"
    ehr_adj_path = './data/MIMIC-III/ehr_adj_final.pkl'
    #molecule_path = "./data/MIMIC-III/atc3toSMILES.pkl"

    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = dill.load(Fin)
    with open(ddi_mask_path, 'rb') as Fin:
        ddi_mask_H = dill.load(Fin)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
    # with open(molecule_path, 'rb') as Fin:
    #     molecule = dill.load(Fin)
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)
    with open(ehr_adj_path, 'rb') as Fin:
        ehr_adj = dill.load(Fin)

    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    )
    
    # 划分数据集
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]

    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")
    
    train_dataset = MimicDataSet(data_train)
    eval_dataset = MimicDataSet(data_eval)
    test_dataset = MimicDataSet(data_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=mimic_collate, shuffle=True, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=mimic_collate, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=mimic_collate, shuffle=True, pin_memory=True)
    
    return data_train, data_eval, data_test, train_dataloader, eval_dataloader, test_dataloader, voc_size, ddi_adj, ddi_mask_H, ehr_adj


def dnntsp_process(med, item_embedding_matrix, device):
    def get_nodes(baskets):
        # convert tensor to int
        # baskets = [basket.tolist() for basket in baskets]
        items = torch.tensor(list(set(itertools.chain.from_iterable(baskets))))
        return items
    def convert_to_gpu(data):
        if device != -1 and torch.cuda.is_available():
            data = data.to(device)
        return data
    def get_edges_weight(baskets):
        edges_weight_dict = defaultdict(float)
        for basket in baskets:
            # basket = basket.tolist()
            for i in range(len(basket)):
                for j in range(i + 1, len(basket)):
                    edges_weight_dict[(basket[i], basket[j])] += 1.0
                    edges_weight_dict[(basket[j], basket[i])] += 1.0
        return edges_weight_dict

    user_data = med
    nodes = get_nodes(baskets=user_data[:-1])

    # 生成嵌入
    nodes_feature = item_embedding_matrix(convert_to_gpu(nodes).long())
    # print(nodes.shape)
    project_nodes = torch.tensor(list(range(nodes.shape[0])))
    # construct fully connected graph, containing N nodes, unweighted
    # (0, 0), (0, 1), ..., (0, N-1), (1, 0), (1, 1), ..., (1, N-1), ...
    # src -> [0, 0, 0, ... N-1, N-1, N-1, ...],  dst -> [0, 1, ..., N-1, ..., 0, 1, ..., N-1]

    src = torch.stack([project_nodes for _ in range(project_nodes.shape[0])], dim=1).flatten().tolist()
    dst = torch.stack([project_nodes for _ in range(project_nodes.shape[0])], dim=0).flatten().tolist()
    g = dgl.graph((src, dst), num_nodes=project_nodes.shape[0])

    # 全部visit的边权重矩阵
    edges_weight_dict = get_edges_weight(user_data[:-1])
    # add self-loop
    for node in nodes.tolist():
        if edges_weight_dict[(node, node)] == 0.0:
            edges_weight_dict[(node, node)] = 1.0
    # normalize weight
    # 正则化
    max_weight = max(edges_weight_dict.values())
    for i, j in edges_weight_dict.items():
        edges_weight_dict[i] = j / max_weight
    # get edge weight for each timestamp, shape (T, N*N)
    # print(edges_weight_dict)
    # 每个子图的一维边权
    edges_weight = []
    for basket in user_data[:-1]:
        # basket = basket.tolist()
        # list containing N * N weights of elements
        edge_weight = []
        for node_1 in nodes.tolist():
            for node_2 in nodes.tolist():
                if (node_1 in basket and node_2 in basket) or (node_1 == node_2):
                    # each node has a self connection
                    edge_weight.append(edges_weight_dict[(node_1, node_2)])
                else:
                    edge_weight.append(0.0)
        edges_weight.append(torch.Tensor(edge_weight))
    # tensor -> shape (T, N*N)
    edges_weight = torch.stack(edges_weight).to(device)
    # 返回全连接图， 节点嵌入， 子图边权， 所有节点列表， 用户数据
    return g, nodes_feature, edges_weight, nodes, user_data

def collate_set_across_user(batch_data, item_total, device):
    def get_truth_data(truth_data):
        truth_list = []
        for basket in truth_data:
            one_hot_items = F.one_hot(torch.tensor(basket), num_classes=item_total)
            one_hot_basket, _ = torch.max(one_hot_items, dim=0)
            truth_list.append(one_hot_basket)
        truth = torch.stack(truth_list)
        return truth

    def convert_to_gpu(data):
        if device != -1 and torch.cuda.is_available():
            data = data.to(device)
        return data
    
    def convert_all_data_to_gpu(*data):
        res = []
        for item in data:
            item = convert_to_gpu(item)
            res.append(item)
        return tuple(res)

    ret = list()
    for idx, item in enumerate(zip(*batch_data)):
        # assert type(item) == tuple
        # 合并所有user的图
        if isinstance(item[0], dgl.DGLGraph):
            ret.append(dgl.batch(item))
        elif isinstance(item[0], torch.Tensor):
            # 处理一维权重
            if idx == 2:
                # pad edges_weight sequence in time dimension batch, (T, N*N)
                # (T_max, N*N)
                max_length = max([data.shape[0] for data in item])
                edges_weight, lengths = list(), list()
                for data in item:
                    if max_length != data.shape[0]:
                        edges_weight.append(torch.cat((data, torch.stack(
                            [torch.eye(int(data.shape[1] ** 0.5)).flatten() for _ in range(max_length - data.shape[0])],
                            dim=0)), dim=0))
                    else:
                        edges_weight.append(data)
                    lengths.append(data.shape[0])
                # (T_max, N_1*N_1 + N_2*N_2 + ... + N_b*N_b)
                ret.append(torch.cat(edges_weight, dim=1))
                # (batch, )
                ret.append(torch.tensor(lengths))
            else:
                # nodes_feature -> (N_1 + N_2, .. + N_b, item_embedding) or nodes -> (N_1 + N_2, .. + N_b, )
                ret.append(torch.cat(item, dim=0))
        # user_data
        elif isinstance(item[0], list):
            data_list = item
        else:
            raise ValueError(f'batch must contain tensors or graphs; found {type(item[0])}')

    truth_data = get_truth_data([dt[-1] for dt in data_list])
    ret.append(truth_data)

    # tensor (batch, items_total), for frequency calculation
    users_frequency = np.zeros([len(batch_data), item_total])
    for idx, baskets in enumerate(data_list):
        for basket in baskets:
            for item in basket:
                users_frequency[idx, item] = users_frequency[idx, item] + 1
    users_frequency = normalize(users_frequency, axis=1, norm='max')
    ret.append(torch.Tensor(users_frequency))


    # (g, nodes_feature, edges_weight, lengths, nodes, truth_data, individual_frequency)
    g, nodes_feature, edges_weight, lengths, nodes, truth_data, individual_frequency = ret
    return convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, individual_frequency)