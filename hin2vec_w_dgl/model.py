# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:19:49 2023

@author: huijian
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np


def binary_reg(x: torch.Tensor):
    a = torch.sigmoid(x)
    b = a.detach()
    c = (x.detach()>0).float()
    return a-b+c

class HIN2Vec(nn.Module):
    def __init__(self,node_size,path_size,emb_dim,sigmoid_reg=False,directed=False):
        super(HIN2Vec,self).__init__()
        self.node_size = node_size
        self.path_size = path_size
        
        self.directed = directed
        self.reg = torch.sigmoid if sigmoid_reg else binary_reg
        
        self.__model_initilization_(node_size, path_size, emb_dim, directed)
        
        return
    
    def __model_initilization_(self,node_size,path_size,emb_dim,directed):
        
        self.start_embeds = nn.Embedding(num_embeddings = node_size, embedding_dim=emb_dim)
        if directed:
            self.end_embeds = nn.Embedding(num_embeddings = node_size, embedding_dim=emb_dim)
        else:
            self.end_embeds = self.start_embeds
        
        self.path_embeds = nn.Embedding(path_size, emb_dim)
        return 
    
    def forward(self, 
                start_node:torch.LongTensor,
                end_node:torch.LongTensor,
                path:torch.LongTensor):
        _s = self.start_embeds(start_node) # (batch_size,embed_size)
        _e = self.end_embeds(end_node) 
        _p = self.path_embeds(path)
        _p = self.reg(_p)
        
        _nodes_pair = torch.mul(_s,_e)
        _r = torch.mul(_nodes_pair,_p)
        
        output = torch.sigmoid(torch.sum(_r,axis=1))
        return output
    
if __name__ == '__main__':
    ## test binary_reg

    print('sigmoid')
    a = torch.tensor([-1.,0.,1.],requires_grad=True)
    b = torch.sigmoid(a)
    c = b.sum()
    print(a)
    print(b)
    print(c)
    c.backward()
    print(c.grad)
    print(b.grad)
    print(a.grad)

    print('binary')
    a = torch.tensor([-1., 0., 1.], requires_grad=True)
    b = binary_reg(a)
    c = b.sum()
    print(a)
    print(b)
    print(c)
    c.backward()
    print(c.grad)
    print(b.grad)
    print(a.grad)
    