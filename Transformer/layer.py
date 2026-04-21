import numpy as np
import torch
import torch.nn as nn
import math
import copy
from clone_layer import clone_layers

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout):
        super().__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer = clone_layers(SublayerConnection(size,dropout),2)
        
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.attn(x,x,x,mask))
        return self.sublayer[1](x, self.feed_forward)
        
class DecoderLayer(nn.Module):
    def __init__(self, size, maskattn, attn, feed_forward, dropout):
        super().__init__()
        self.maskattn = maskattn
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer = clone_layers(SublayerConnection(size, dropout),3)
    
    def forward(self, x, encode, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x:self.maskattn(x,x,x,tgt_mask))
        x = self.sublayer[1](x, lambda x:self.attn(x,encode,encode,src_mask))
        return self.sublayer[2](x, self.feed_forward)
        
def subsequent_mask(size):
    "Mask out subsequent positions. The position which is true are valid positions"
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0