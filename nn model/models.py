
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.max_ = None
        self.emb_dropout = torch.nn.Dropout(p=0.2)
        self.seq_len = 2
        

    def seq_embed(self, input_ids, all_ids =None, index=None, seq_len=2):
        if seq_len != 0:
            input_embed = self.emb_dropout(self.item_emb(input_ids[:,-seq_len:]))
        else:
            mask = input_ids.clone()
            mask[mask != 0] = 1
            input_embed = self.emb_dropout(self.item_emb(input_ids[:,-seq_len:])) * mask.unsqueeze(-1)
            input_embed = input_embed.sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)
        input_embed = input_embed.reshape(input_ids.shape[0],-1)
        
        # input_embed_GRU = self.emb_dropout(self.item_emb(all_ids))
        # input_embed_GRU, _ = self.gru(input_embed_GRU)
        # input_embed_GRU = input_embed_GRU.reshape(-1,input_embed_GRU.shape[-1])[index,:]
        # input_embed = torch.cat([input_embed,input_embed_GRU],dim=-1)
        input_embed = self.mlp(input_embed)
        input_embed = F.normalize(input_embed, p=2, dim=1)
        return input_embed
    
    def recall_embed(self, recall_ids):
        item_embed = self.emb_dropout(self.item_emb(recall_ids))
        item_embed = F.normalize(item_embed, p=2, dim=1)
        return item_embed
    
    def forward(self,
                input_ids,
                recall_items,
                all_ids = None,
                index = None,
                number = 0,
               ):
        input_embed = self.seq_embed(input_ids, all_ids, index, self.seq_len)
        item_embed = self.recall_embed(recall_items)
        
        logits = input_embed.mm(item_embed.T)
        logits = torch.softmax(logits,dim=1)
        if number != 0:
            temp = torch.arange(number, number+logits.shape[1], 1, device=self.dev)
            logits = logits + 1/(temp*temp)
        
        return logits
    
    def get_loss(self,
                 input_ids,
                 label,
                 all_ids=None,
                 index = None,
                 temp=0.05
                ):
        input_embed = self.seq_embed(input_ids, all_ids, index, self.seq_len)
        item_embed = self.recall_embed(label)
        
        y_pred = torch.cat([input_embed,item_embed],dim=0)
        y_true = torch.arange(y_pred.shape[0], device=self.dev)
        y_true = (y_true + input_embed.shape[0]) % y_pred.shape[0]
        sim = y_pred.mm(y_pred.T)
        sim = sim - torch.eye(y_pred.shape[0], device=self.dev) * 1e12
        sim = sim / temp
        loss = F.cross_entropy(sim, y_true)
        

        return loss

    def reset_para(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.item_emb.weight, std=0.01)
                
class DoubleSeq(BaseModel):
    def __init__(self, item_num, args, path=None):
        super(DoubleSeq, self).__init__()
        
        self.max_ = args.max_
        self.item_num = item_num
        self.dev = args.device
        self.seq_len = args.seq_len
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        # self.gru = nn.GRU(input_size=128, hidden_size=128)
        if self.max_ != None:
            self.mlp = nn.Sequential(
                torch.nn.Linear(args.hidden_units*4, args.hidden_units),
                torch.nn.GELU(),
                torch.nn.Linear(args.hidden_units, args.hidden_units)
            )
        else:
            self.mlp = nn.Sequential(
                torch.nn.Linear(args.hidden_units*2, args.hidden_units),
                torch.nn.GELU(),
                torch.nn.Linear(args.hidden_units, args.hidden_units)
            )
        self.reset_para()
        if path is not None:
            self.item_emb = torch.load(path)

    
class TripleSeq(BaseModel):
    def __init__(self, item_num, args, path=None):
        super(TripleSeq, self).__init__()

        self.item_num = item_num
        self.dev = args.device
        self.seq_len = 3
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.mlp = nn.Sequential(
                torch.nn.Linear(args.hidden_units*3, args.hidden_units),
                torch.nn.GELU(),
                torch.nn.Linear(args.hidden_units, args.hidden_units)
            )
        self.reset_para()
        if path is not None:
            self.item_emb = torch.load(path)
        
class WeightSeq(torch.nn.Module):
    def __init__(self, item_num, args, path=None):
        super(WeightSeq, self).__init__()

        self.item_num = item_num
        self.dev = args.device
        self.seq_len = 0
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.max_ = None
        self.emb_dropout = torch.nn.Dropout(p=0.2)
        self.seq_len = 2
        self.mlp = nn.Sequential(
                torch.nn.Linear(args.hidden_units*4, args.hidden_units*4),
                torch.nn.GELU(),
                torch.nn.Linear(args.hidden_units*4, args.hidden_units)
            )
        self.reset_para()
        if path is not None:
            self.item_emb = torch.load(path)


    def seq_embed(self, input_ids, all_ids =None, index=None, seq_len=2):
        input_embed = self.emb_dropout(self.item_emb(input_ids)) 
        a = input_embed[:,:-4,:]
        mask = input_ids.clone()[:,:-4]
        mask[mask!=0] = 1
        a = (a*mask.unsqueeze(-1)).sum(dim=-2)
        mask = mask.sum(dim=-1).unsqueeze(-1)
        mask_t = mask.clone()
        mask_t[mask_t==0]=1
        a = a/mask_t
        mask[mask==0]=-1
        mask[mask>0]=0
        mask[mask==-1]=1
        add = self.emb_dropout(self.item_emb(torch.tensor([0]).cuda())).repeat(input_ids.shape[0],1)
        add = add *mask
        a = a+add
        
        b = input_embed[:,-4:-2,:]
        mask = input_ids.clone()[:,-4:-2]
        mask[mask!=0] = 1
        b = (b*mask.unsqueeze(-1)).sum(dim=-2)
        mask = mask.sum(dim=-1).unsqueeze(-1)
        mask_t = mask.clone()
        mask_t[mask_t==0]=1
        b = b/mask_t
        mask[mask==0]=-1
        mask[mask>0]=0
        mask[mask==-1]=1
        add = self.emb_dropout(self.item_emb(torch.tensor([0]).cuda())).repeat(input_ids.shape[0],1)
        add = add *mask
        b = b+add
        
        c = input_embed[:,-2,:]

        d = input_embed[:,-1,:]
        
        input_embed = self.mlp(torch.cat([a,b,c,d],dim=-1))
        input_embed = F.normalize(input_embed, p=2, dim=1)
        return input_embed
    
    def recall_embed(self, recall_ids):
        item_embed = self.emb_dropout(self.item_emb(recall_ids))
        item_embed = F.normalize(item_embed, p=2, dim=1)
        return item_embed
    
    def forward(self,
                input_ids,
                recall_items,
                all_ids = None,
                index = None,
                number = 0,
               ):
        input_embed = self.seq_embed(input_ids, all_ids, index, self.seq_len)
        item_embed = self.recall_embed(recall_items)
        
        logits = input_embed.mm(item_embed.T)
        logits = torch.softmax(logits,dim=1)
        if number != 0:
            temp = torch.arange(number, number+logits.shape[1], 1, device=self.dev)
            logits = logits + 1/(temp*temp)
        
        return logits
    
    def get_loss(self,
                 input_ids,
                 label,
                 all_ids=None,
                 index = None,
                 temp=0.05
                ):
        input_embed = self.seq_embed(input_ids, all_ids, index, self.seq_len)
        item_embed = self.recall_embed(label)
        
        y_pred = torch.cat([input_embed,item_embed],dim=0)
        y_true = torch.arange(y_pred.shape[0], device=self.dev)
        y_true = (y_true + input_embed.shape[0]) % y_pred.shape[0]
        sim = y_pred.mm(y_pred.T)
        sim = sim - torch.eye(y_pred.shape[0], device=self.dev) * 1e12
        sim = sim / temp
        loss = F.cross_entropy(sim, y_true)
        

        return loss

    def reset_para(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.item_emb.weight, std=0.01)
