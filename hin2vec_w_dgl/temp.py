# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 13:42:22 2023

@author: huijian
"""

import dgl
import torch
import numpy as np
import pickle

"""
g2 = dgl.heterograph({
    ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]),
    ('user', 'view', 'item'): ([0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 2, 1]),
    ('item', 'viewed-by', 'user'): ([0, 1, 1, 2, 2, 1], [0, 0, 1, 2, 3, 3])})


metapath1 = ['follow', 'view', 'viewed-by']
metapath2 = [ 'view','viewed-by',"follow"]

walks= dgl.sampling.random_walk(
        g2, [0, 1, 2, 3], metapath=metapath2)

print(walks)
"""

if __name__=="__main__":
    print("load the model and get the np.array")
    model = torch.load("embed.pkl")
    embed = model.start_embeds
    params = list(embed.parameters())[0]
    params = params.cpu().detach().numpy()
    # hin2vec_embed = pickle.dumps(params)
    
    with open("hin2vec_emb.pkl","wb") as _file:
        pickle.dump(params,_file)
    with open("hin2vec_emb.pkl","rb") as _file:
        restored_params = pickle.load(_file)