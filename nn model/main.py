import os
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter
from torch.cuda.amp import autocast, GradScaler
from torch.optim import *
from torch.utils.data import Dataset, DataLoader
from lion import Lion
from sklearn.model_selection import train_test_split
from models import *
from utils import *
from evaluate import evaluate
from test import test
device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    
    locale2ids = pickle.load(open('../data/local2ids.pkl', 'rb'))
    train_data = pd.read_pickle('../data/train_data_plus_new.dataset')
    valid_data = pd.read_pickle('../data/valid_data_005.dataset')
    
    print(len(train_data),len(valid_data))
    itemnum = 1410675
    
    train_set = MyDataset(train_data)
    valid_set = MyDataset(valid_data)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn_train, shuffle=True, pin_memory=True,
                                  num_workers=6)
    valid_dataloader = DataLoader(valid_set, batch_size=1, collate_fn=collate_fn_valid, shuffle=False,pin_memory=False,
                                  num_workers=2)
    
    model = WeightSeq(itemnum, args, path='embedding.pth').to(device)
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.95, 0.98), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=1,gamma = 0.33)
    
    best_mrr = 0
    best_epo = -1
    os.makedirs('ckpt', exist_ok=True)
    model_save_path = 'ckpt/task1-weight_embd2.pt'
    if args.train is 1:
        print('----------------Training-Step1----------------')
        for epoch in range(args.epochs):
            # break
            loss_avg = AverageMeter()
            for step, [input_ids, labels, all_ids, index] in enumerate(tqdm(train_dataloader)):
                model.train()
                with autocast():
                    loss = model.get_loss(input_ids.to(device),
                                          labels.to(device),
                                          all_ids.to(device),
                                          index.to(device))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
                loss_avg.update(loss.item())

            print(f'Epoch [{epoch + 1} / {args.epochs}]: loss={loss_avg.avg}')

            if epoch % 1 == 0 :
                model.eval()
                mrr, ndcg, ht = evaluate(model, valid_dataloader, locale2ids, device, args)
                print('Epoch:%d, MRR@10: %.4f, NDCG@10: %.4f, HR@10: %.4f' % (epoch + 1, mrr, ndcg, ht))
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_epo = epoch
                    torch.save(model.state_dict(), model_save_path)
                else:
                    if epoch - best_epo >= args.patience:
                        model.load_state_dict(torch.load(model_save_path))
                        if optimizer.param_groups[0]['lr'] > 1e-4:
                            scheduler.step()
                        else:
                            break
    else:
        model.load_state_dict(torch.load('ckpt/' + args.model_path))
    model.eval()
    mrr, ndcg, ht = evaluate(model, valid_dataloader, locale2ids, device, args)
    print('MRR@10: %.4f, NDCG@10: %.4f, HR@10: %.4f' % (mrr, ndcg, ht))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--patience', default=1, type=int)
    parser.add_argument('--lion', default=0, type=int)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--save_number', default=1, type=int)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--model_path', default='0.2642.pt', type=str)
    parser.add_argument('--max_', default=None, type=str)
    parser.add_argument('--seq_len', default=2, type=int)

    args = parser.parse_args()

    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    seed_all(args.seed)

    main(args)
