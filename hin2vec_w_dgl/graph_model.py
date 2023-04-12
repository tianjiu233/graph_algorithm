# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 22:02:46 2023

@author: huijian
"""

import torch
import torch.nn
from torch.utils.data import Dataset,DataLoader
import dgl

import random
import numpy as np

# mylib
from build_graph import build_hete_graph
from build_graph import generate_metapath
from model import HIN2Vec



# ------ aux fcns(begin) ------
def find_all_w_hop(w_hop,edge_path_dict,flag="-"):
    
    # 辅助函数，获得的relations将用于采样时，对应w_hop之内可能存在的所有关系
    relations = []
    for hop in range(w_hop):
        new_relations = []
        if hop == 0:
            for r_edge in list(edge_path_dict.keys()):
                new_relations.append(str(r_edge))
        else:
            # new_relations = []
            for _cur_r in relations: # _cur_r : str
                tmp_r = _cur_r.split(flag)[-1]
                tmp_r = int(tmp_r)
                for _next_r in edge_path_dict[tmp_r]: # _next_r : int
                    _r = _cur_r + flag + str(_next_r)
                    new_relations.append(_r)
                
        # update the relations
        relations = new_relations
    
    return relations

class TrainDatasetwNegativeSample(Dataset):
    def __init__(self,sample,node_size,neg=5):
        """
        param node_size: 节点数目
        param neg: 负采样数目
        param sample: [(start_node,end_node,path_id),...,]
        """
        print("build (batch) training dataset...")
        
        _num = len(sample)
        x = np.tile(sample,(neg+1,1)) # 扩展数目为1->1*(neg+1)倍
        y = np.zeros(_num*(1+neg))
        y[:_num] = 1
        
        # 进行随机的替换
        x[_num:,1] = np.random.randint(0,node_size-1,(_num*neg,))
        
        self.x = torch.LongTensor(x)
        self.y = torch.FloatTensor(y)
        self.length = len(x)
        
        return
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return self.length


# ------ aux fcns(end) ------


class HIN2Vec_w_Graph(object):
    def __init__(self,data_dir,edge_path_dict,emb_dim=128,w_hop=2,metapath_length=5,flag="-"):
        
        # step-1.构建图数据
        hete_graph,raw_data = build_hete_graph(data_dir)
        
        self.graph = hete_graph
        self.raw_data = raw_data
        self.edge_path_dict = edge_path_dict # 这个需要自己总结
        
        self.flag = flag
        
        self.metapath_length = metapath_length
        self.w_hop = w_hop
        relations = []
        for i in range(1,w_hop+1):
            tmp = find_all_w_hop(w_hop=i,edge_path_dict=self.edge_path_dict,flag=self.flag)
            relations = relations + tmp
        # 为在within w_hop之内所需要考虑的所有relations
        self.relations = relations
        
        # 构建metapath的映射关系
        path_idx2relation_dict = {}
        relation2path_idx_dict = {}
        idx_key = 0
        for _r in relations:
            path_idx2relation_dict[idx_key] = _r
            relation2path_idx_dict[_r] = idx_key
            idx_key = idx_key + 1
            
        self.path_idx2relation_dict = path_idx2relation_dict
        self.relation2path_idx_dict = relation2path_idx_dict
        
        # step-2.构建模型
        self.emb_dim = emb_dim
        self.node_size = raw_data.nodes["total"]
        self.path_size = len(relations)
        self.model = HIN2Vec(node_size=self.node_size, path_size=self.path_size, emb_dim=self.emb_dim)
        
        return 
    
    def random_walk_sample_w_start_nodes(self,nodes):
        # 为了确保遍历所有的点，则以其起点为开始，通过起点确定对应的metapath，并生成采样的trace
        return
    
    def random_walk_sample(self,batch_size,metapth=None):
        
        # 1.获取本次随机采样的metapth,如果没有指定，则调用函数随机生成
        if metapth is None:
            metapath = generate_metapath(self.metapath_length,self.edge_path_dict)
        
        sampled_relations_list = []
        flag = "-"
        for sampled_node_idx in range(self.metapath_length+1 - self.w_hop):
            _bias = sampled_node_idx
            for i in range(self.w_hop):
                if i == 0:
                    _r = metapath[_bias]
                    _r = str(_r)
                else:
                    _r = _r + flag + str(metapath[_bias+i])
                sampled_relations_list.append(_r)
        # 将sampled_relations_list映射到sampled_path_idx_list
        def _map(_r):
            path_idx_list = self.relation2path_idx_dict[_r]
            return path_idx_list
        # 这里的path_idx对应我们进行hin2vec训练时的元路径id
        sampled_path_idx_list = list(map(_map,sampled_relations_list))
        
        # 2.采样！
        node_type = self.raw_data.load_links()["meta"][metapath[0]][0]
        _typed_node_count = self.raw_data.nodes["count"][node_type]
        
        # 这里我们对首节点采取的是随机采样获取的，而非遍历
        start_nodes = [random.randint(0,_typed_node_count-1) for _ in range(batch_size)] # list
        # traces: torch.int32 (len(start_nodes,metapath_length+1))
        traces,sampled_nodes_types = dgl.sampling.random_walk(g = self.graph, 
                                                              nodes = start_nodes,
                                                              metapath=metapath)
        
        # 3.构建训练样本
        train_data = []
        traces = traces.cpu().detach().numpy() # np.array
        sampled_nodes_types = sampled_nodes_types.cpu().detach().numpy() # np.array
        _shift_dict = self.raw_data.nodes["shift"]
        for _t in traces:
            # process every traces
            for i in range(self.metapath_length+1-self.w_hop):
                typed_start_node_idx = _t[i]
                # 在start_node_idx处进行break是没有意义的
                # if typed_start_node_idx < 0:
                  #  break
                start_node_type = sampled_nodes_types[i]
                # global idx
                start_node_idx = typed_start_node_idx + _shift_dict[start_node_type]
                for j in range(1,self.w_hop+1): # 1,2,w_hop
                    typed_end_node_idx = _t[i+j]
                    if typed_end_node_idx < 0:
                        break
                    end_node_type = sampled_nodes_types[i+j]
                    # global idx
                    end_node_idx = typed_end_node_idx + _shift_dict[end_node_type]
                    _path_idx = sampled_path_idx_list[i*self.w_hop + j - 1]
                    train_sample = [start_node_idx,end_node_idx,_path_idx]     
                    train_data.append(train_sample)
        
        # 此时返回的train_data应该只是一个嵌套的list
        return train_data,traces,sampled_nodes_types
    
    def build_train_dataset(self,batch_size,neg=5,metapath=None):
        
        train_data,traces,sampled_nodes_types = self.random_walk_sample(batch_size,metapath)

        # list -> np.array
        train_data = np.array(train_data,dtype=np.int32)
        
        train_dataset = TrainDatasetwNegativeSample(sample=train_data,node_size=self.node_size,neg=neg)
        
        
        return train_dataset
    
    
if __name__=="__main__":
    print("Test graph_model.py")
    
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
    
    model_w_graph = HIN2Vec_w_Graph(data_dir,edge_path_dict,
                                    emb_dim=128,
                                    w_hop=2,
                                    metapath_length=5,
                                    flag="-")
    train_data,traces,sampled_nodes_types = model_w_graph.random_walk_sample(batch_size=32)
    train_dataset = model_w_graph.build_train_dataset(batch_size=32,neg=5)