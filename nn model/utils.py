import os
import time
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torch.nn.functional as F


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, fmt):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=fmt)


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MyDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        data = self.df.iloc[idx]

        return data


def collate_fn_train(data):
    input_ids, labels, all_ids, index = [], [], [], []
    max_len = max(6,max([len(x['session']) for x in data]))
    for x in data:
        labels.append(x['session'][-1])
        session =  x['session'][:-1]
        session = [0]*(max_len-len(session)) +session
        input_ids.append(session)
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    all_ids = torch.tensor(all_ids)
    index = torch.tensor(index)
    return input_ids, labels, all_ids, index




def collate_fn_valid(data):
    input_ids, labels, all_ids, index = [], [], [], []
    for x in data:
        labels = x['session'][-1]
        session =  x['session'][:-1]
        
        session = [0,0,0,0,0] + list(dict.fromkeys(session[::-1]))[::-1]
        session = session
        input_ids.append(session)
        all_ids.append(session)
        index.append(len(session) -1 )
    locale = x['locale']
    input_ids = torch.tensor(input_ids)
    all_ids = torch.tensor(all_ids)
    index = torch.tensor(index)
    return input_ids, labels, locale, all_ids, index
