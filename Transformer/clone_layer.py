import copy
import torch
import torch.nn as nn

def clone_layers(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
