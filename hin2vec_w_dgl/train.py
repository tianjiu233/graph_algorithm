# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:57:23 2023

@author: huijian
"""
from torch.utils.data import Dataset,DataLoader

def train_model(log_interval,model,device,train_loader:DataLoader,optimizer,loss_function,epoch):
    
    model.train()
    
    for idx,(_data,_label) in enumerate(train_loader):
        _data,_label = _data.to(device),_label.to(device)
        optimizer.zero_grad()
        _output = model(_data[:,0],_data[:,1],_data[:,2])
        loss = loss_function(_output.view(-1),_label)
        loss.backward()
        optimizer.step()
    
    if idx % log_interval ==0:
        print(f'\rTrain Epoch: {epoch} '
                  f'[{idx * len(_data)}/{len(train_loader.dataset)} ({100. * idx / len(train_loader):.3f}%)]\t'
                  f'Loss: {loss.item():.3f}\t\t',
                  end='')
    
    return