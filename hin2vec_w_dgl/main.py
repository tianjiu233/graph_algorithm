# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 23:51:52 2023

@author: huijian
"""

import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import random
import numpy as np
from tqdm import tqdm

# mylib
from build_graph import build_hete_graph
from build_graph import generate_metapath
from graph_model import HIN2Vec_w_Graph


from torch.utils.data import Dataset,DataLoader
def train(log_interval, model, device, train_loader: DataLoader, optimizer, loss_function, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data[:, 0], data[:, 1], data[:, 2])
        loss = loss_function(output.view(-1), target)
        loss.backward()
        optimizer.step()
    
    # if idx % log_interval == 0:
    print(f'\rTrain Epoch: {epoch} '
          f'[{idx * len(data)}/{len(train_loader.dataset)} ({100. * idx / len(train_loader):.3f}%)]\t'
          f'Loss: {loss.item():.3f}\t\t',
          # f'data = {data}\t target = {target}',
          end='')   
        
    # print()


if __name__=="__main__":
    
    print("Testing hin2vec algorithm")
    
    # 1.构建图以及相应的模型
    data_name = "DBLP"
    data_dir = f'./nc_data/{data_name}'
    edge_path_dict = {
        0:[1,2],
        3:[0],
        1:[4],
        2:[5],
        4:[1,2,3],
        5:[1,2,3],
        }
    
    metapath_length = 128
    emb_dim = 128
    model_w_graph = HIN2Vec_w_Graph(data_dir,edge_path_dict,
                                    emb_dim=emb_dim,
                                    w_hop=2,
                                    metapath_length= metapath_length,
                                    flag="-")
    
    
    # 2.训练使用的超参数
    optimizer = optim.AdamW(model_w_graph.model.parameters())
    loss_function = nn.BCELoss()
    device = torch.device("cpu")
    batch_size = 64
    epochs = 500
    log_interval = 10
    print("begin train the model")
    for _e in tqdm(range(epochs)):
        train_dataset = model_w_graph.build_train_dataset(batch_size=batch_size)
        data_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
        train(log_interval,model=model_w_graph.model,device=device,train_loader=data_loader,optimizer=optimizer,loss_function=loss_function,epoch=_e)
    