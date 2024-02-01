
import torch
import numpy as np
from util import llprint
from metrics import multi_label_metric, ddi_rate_score
from torch.optim import Adam
from collections import defaultdict
import time
import dill
import torch.nn.functional as F
import os
from util import *


def Test(model, model_path, device, data_test, voc_size, drug_data):
    with open(model_path, 'rb') as Fin:
        model.load_state_dict(torch.load(Fin, map_location=device))
    model = model.to(device).eval()
    print('--------------------Begin Testing--------------------')
    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
    tic, result, sample_size = time.time(), [], round(len(data_test) * 0.8)
    np.random.seed(0)
    for _ in range(10):
        test_sample = np.random.choice(data_test, sample_size, replace=True)
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = \
            eval_1batch(model, test_sample, voc_size)
        result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
    result = np.array(result)
    mean, std = result.mean(axis=0), result.std(axis=0)
    metric_list = ['ddi_rate', 'ja', 'avg_f1', 'prauc', 'med']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])
    print(outstring)
    print('average test time: {}'.format((time.time() - tic) / 10))
    print('parameters', get_n_params(model))

def eval_1batch(model, data_eval, voc_size):
    model.eval()  # 将模型设置为评估模式

    smm_record = []  # 每个用户的推荐药物的下标记录
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]  # 初始化列表，用于存储各个指标的计算结果
    med_cnt, visit_cnt = 0, 0  # 预测的药物总数和访问总数，用于计算平均药物数

    for step, input in enumerate(data_eval):  # 遍历数据集中的每个样本
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []  # 初始化列表，用于存储每个样本的真值、预测结果、预测概率和预测标签
        skip_num = 0
        flag = 1

        for adm_idx, adm in enumerate(input):  # 遍历每个样本的序列
            if adm_idx == 0 or len(input[adm_idx-1][2]) < 2:
                skip_num += 1
                continue
            flag = 0
            target_output, _ = model(input[: adm_idx + 1])  # 使用模型进行预测

            # 构建medicine的真值的多热编码向量
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # 预测概率
            target_output = torch.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)
            

            # 将预测概率向量转换为多热编码向量
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # 预测标签 取行索引
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1  # 总访问次数
            med_cnt += len(y_pred_label_tmp)  # 预测的药物数量

        if flag:
            continue
        smm_record.append(y_pred_label)  # 记录每个用户的推荐药物的下标

        # 计算同一用户多次访问的多个标签指标的平均值
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\reval step: {} / {}".format(step, len(data_eval)))  # 打印当前步骤和总步骤数


    # 计算DDI率
    ddi_rate = ddi_rate_score(smm_record, path="./data/MIMIC-III/ddi_A_final.pkl")


    # 计算所有样本的平均值
    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


def Test(model, model_path, device, data_test, voc_size):
    with open(model_path, 'rb') as Fin:
        model.load_state_dict(torch.load(Fin, map_location=device))
    model = model.to(device).eval()
    print('--------------------Begin Testing--------------------')
    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
    tic, result, sample_size = time.time(), [], round(len(data_test) * 0.8)

    np.random.seed(0)
    for _ in range(10):
        test_sample = np.random.choice(range(len(data_test)), sample_size, replace=True)
        test_sample = [data_test[i] for i in test_sample]
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = \
            eval_1batch(model, test_sample, voc_size)
        result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
    result = np.array(result)
    mean, std = result.mean(axis=0), result.std(axis=0)
    metric_list = ['ddi_rate', 'ja', 'avg_f1', 'prauc', 'med']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])
    print(outstring)
    print('average test time: {}'.format((time.time() - tic) / 10))
    print('parameters', get_n_params(model))




def train_1batch(model, device, data_train, data_eval, voc_size, args):
    model.to(device=device)
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0
    run_num=0
    EPOCH = 200
    for epoch in range(EPOCH):
        tic = time.time()
        print(f'----------------Epoch {epoch}------------------')
        skip_num = 0
        model.train()
        for step, input in enumerate(data_train):
            
            loss = 0
            for idx, adm in enumerate(input):
                if idx == 0 or len(input[idx-1][2]) < 2:
                    skip_num += 1
                    continue
                run_num += 1
                # medicine的二维multi-hot编向量，用来计算bce损失
                loss_bce_target = torch.zeros((1, voc_size[2])).to(device)
                loss_bce_target[:, adm[2]] = 1

                # medicine的二维multi-label向量，用来计算mlm损失
                loss_multi_target = -torch.ones((1, voc_size[2])).long()
                for id, item in enumerate(adm[2]):
                    loss_multi_target[0][id] = item
                loss_multi_target = loss_multi_target.to(device)

                # 运行模型
                result, loss_ddi = model(input[: idx + 1]) # 输入第0~i次访问的数据
                #print('result', result)

                # loss函数定义
                loss_bce = F.binary_cross_entropy_with_logits(result, loss_bce_target)
                loss_multi = F.multilabel_margin_loss(F.sigmoid(result), loss_multi_target)

                # multi-hot结果处理
                result = F.sigmoid(result).detach().cpu().numpy()[0] # sigmoid
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score(
                    [[y_label]], path="./data/MIMIC-III/ddi_A_final.pkl"
                )
                # loss = 0.95 * loss_bce + 0.05 * loss_multi

                # 多loss合并，如果当前ddi小于目标ddi，就不计算ddi loss
                if current_ddi_rate <= args.target_ddi:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi
                else:
                    beta = min(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp)
                    loss = (
                        beta * (0.95 * loss_bce + 0.05 * loss_multi)
                        + (1 - beta) * loss_ddi
                    )
                if run_num % 1000 == 0:
                    print(f'loss:{epoch}-{run_num}-{loss}')

                # 梯度计算
                optimizer.zero_grad()
                loss.backward() # retain_graph=True
                #print(f'loss:{step}-{idx}-{loss}')
                optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(data_train)))

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval_1batch(
            model, data_eval, voc_size
        )
        print(
            "training time: {}, validate time: {}".format(
                tic2 - tic, time.time() - tic2
            )
        )
        print(f'skip num:{skip_num}')

        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)

        # 每个epoch的metrics
        if epoch >= 5:
            print(
                "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                    np.mean(history["ddi_rate"][-5:]),
                    np.mean(history["med"][-5:]),
                    np.mean(history["ja"][-5:]),
                    np.mean(history["avg_f1"][-5:]),
                    np.mean(history["prauc"][-5:]),
                )
            )


        torch.save(model.state_dict(), os.path.join("ablation", "2Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model".format(epoch, args.target_ddi, ja, ddi_rate)))

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print("best_epoch: {}".format(best_epoch))

    dill.dump(history,open(os.path.join("saved", "history.pkl""wb")))