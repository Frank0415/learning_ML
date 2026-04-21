import numpy as np
import torch
import torch.nn as nn
import math
import copy
from clone_layer import clone_layers

def attention(q,k,v, mask = None, dropout = None):
    d_k = q.size(-1)
    scores = torch.matmul(q , k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, v), p_attn
    
class MultiHeadAttn(nn.Module):
    def __init__(self, head, d_model, dropout = 0.1):
        super().__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.linears = clone_layers(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, q, k, v, mask = None):
        if mask is not None:
            mask.unsqueeze(1)
        nbatches = q.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        q, k, v = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (q, k, v))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)

        # 3) Concat using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del q
        del k
        del v
        return self.linears[-1](x)