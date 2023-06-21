import sys
import copy
import random
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import pandas as pd

def evaluate(model, dataloader, locale2ids, device, args):
    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    MRR = 0.0
    lens = len(dataloader)
    recall = []
    locales = []
    a = 0
    for batch_idx, [input_ids, labels, locale, all_ids, index_] in enumerate(dataloader):
        print('\r', batch_idx, '/', lens, end='')
        valid_user += 1
        if valid_user == 5000:
            break
        all_items = locale2ids[locale]
        recall_items = torch.tensor(all_items).to(device)
        with torch.no_grad():
            with autocast():
                pred = model(input_ids.to(device),
                             recall_items,
                             all_ids.to(device),
                             index_.to(device),
                             number=0)[0]
        idx = pred.topk(200).indices 
        pred_items_ids = recall_items[idx.cpu().numpy()].tolist()
        if labels not in pred_items_ids:
            continue
        rank = 999
        number = 0
        for item in pred_items_ids:
            if item in all_ids[0]:
                continue
            number+=1
            if number>100:
                break
            if item == labels:
                rank = number
                break

        if rank < 100:
            NDCG += 1 / np.log2(rank + 1)
            HT += 1
            MRR += 1 / (rank)
    return MRR / 5000, NDCG / 5000, HT / 5000
