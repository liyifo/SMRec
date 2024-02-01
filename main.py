# DPM2M框架

import torch
import argparse
from load_data import *
from train import *
from model import *

# Parameters setting
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", default=False, help="test mode")
parser.add_argument("--resume_path", type=str, default='./saved/4Epoch_4_TARGET_0.06_JA_0.5313_DDI_0.06986.model', help="resume path")
parser.add_argument("--device", type=int, default=1, help="gpu id to run on, negative for cpu")

parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--epoch', type=int, default=50, help='# of epoches')
parser.add_argument('--seed', type=int, default=1029, help='random seed')

parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument('--dp', default=0.3, type=float, help='dropout ratio')
parser.add_argument("--dim", type=int, default=64, help="dimension")
parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimension size')
parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")


args = parser.parse_args()


torch.manual_seed(1203)
np.random.seed(2048)

def get_model_name(args):
    model_name = [
        f'dim_{args.emb_dim}',  f'lr_{args.lr}', f'coef_{args.kp}',
        f'dp_{args.dp}', f'ddi_{args.target_ddi}'
    ]
    if args.embedding:
        model_name.append('embedding')
    return '-'.join(model_name)

# run framework
def main():
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')


    # load data
    data_train, data_eval, data_test, train_dataloader, eval_dataloader, test_dataloader, voc_size, ddi_adj, ddi_mask_H, ehr_adj = load_mimic3(device, args)

    # load model
    model = MainModel_level_no_share_new(device, voc_size, args.emb_dim, ehr_adj, ddi_adj, args.dp).to(device)
    if args.test:
        Test(model, args.resume_path, device, data_test, voc_size)
    else:
        train_1batch(model, device, data_train, data_eval, voc_size, args)




if __name__ == '__main__':
    main()

