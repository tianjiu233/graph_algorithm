# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:13:26 2023

@author: huijian
"""

import random
import pandas as pd
from collections import defaultdict
import dgl

# for test
from dblp_data import data_loader_nc_lp,data_loader
import torch

"""
def build_nodes_lut(raw_data):
    # 获取图中点的数目信息
    nodes_num = raw_data.nodes["total"]
    
    lut_idx2type_idx = {}
    lut_type_idx2idx = {}
    
    # 
    _type_count_dict = {}
    _types = raw_data.types["types"]
    for _t in _types:
        _type_count_dict[_t] = 0 # init
    
    for _t in nodes_type: # nodes_type:list
        for idx in range(nodes_num): # from 0 to nodes_num-1
            _type = raw_data.get_node_type(idx)
            # get the type_idx
            type_idx = _type_count_dict[_type]
            
            # update the lut
            lut_idx2type_idx[idx] = type_idx
            _key = str(_type) + "_" + str(type_idx) # special key for lut_type_idx2idx
            lut_type_idx2idx[_key] = idx
            
            # update
            _type_count_dict[_type] = type_idx + 1
    
    # _type_count_dict可以用来验证是否正确构造了相关的lut
    return lut_idx2type_idx,lut_type_idx2idx,_type_count_dict

# 用来适配
def idx2type_idx(idx_list,lut_idx2type_idx):
    
    def map_dict(x):
        return lut_idx2type_idx[x]
    
    new_idx_list = map(map_dict,idx_list)
    
    return new_idx_list
"""

def idx2type_idx(idx_list,raw_data):
    shift_dict = raw_data.nodes["shift"]
    def map_idx(idx):
        _type = raw_data.get_node_type(idx)
        idx2type = idx - shift_dict[_type]
        return idx2type
    new_idx_list = list(map(map_idx,idx_list))
    return new_idx_list


def build_hin(data_dir):
    raw_data = data_loader_nc_lp(data_dir)
    # nodes
    nodes_data = raw_data.nodes # dict
    _shift_list = raw_data.nodes["shift"]
    nodes_type = raw_data.types["data"]  # raw_data.types["types"]
    links_data = raw_data.load_links()
    links_mat = links_data["data"] # dict
    _meta = links_data["meta"]
    _meta_keys = list(_meta.keys()) # 0,1,2,3,4...
    
    # build the data_dict for dgl.heterogenous_graph
    _data_dict_4_hin = {}
    # links_data["meta"]中每一个key都代表了其对应的关系即（）
    for _r in _meta_keys:
        # 确定当前边类型
        _path_type = (_meta[_r][0],_r,_meta[_r][1]) # start_type,path_type,end_type
        # 获取对应的关系矩阵
        _mat = links_mat[_r]
        _rows,_cols = _mat.nonzero()
        _s = torch.tensor(_rows)
        _e = torch.tensor(_cols)
        
        _data_dict_4_hin[_path_type] = (_s,_e)
    
    _graph = dgl.heterograph(_data_dict_4_hin)
    return _graph,raw_data


if __name__=="__main__":
    
    # 
    data_name = "DBLP"
    data_dir = f'./nc_data/{data_name}'
    
    _graph,raw = build_hin(data_dir)
    
    _meta_path = [(0, 1, 0), (1, 2, 1), (1, 3, 2), (1, 0, 3), (2, 1, 4), (3, 1, 5)]
    # 在_graph的基础上进行随机游走
    dgl.sampling.random_walk(g = _graph, 
                             nodes = [0,1,2,3],
                             metapath,
                             length)
    
    
    
    # build the data
    """
    raw_data = data_loader_nc_lp(data_dir)
    # raw_data = data_loader(data_dir)
    
    # 以下是对应的函数代码
    dest_node = []
    source_node = []
    weight = []
    source_class = []
    dest_class = []
    edge_class = []
    
    # nodes
    nodes_data = raw_data.nodes # dict
    nodes_type = raw_data.types["data"]  # raw_data.types["types"]
    links_data = raw_data.load_links()
    links_mat = links_data["data"] # dict
    _meta = links_data["meta"]
    _meta_keys = list(_meta.keys()) # 0,1,2,3,4...
    
    
    # build the data_dict for dgl.heterogenous_graph
    _data_dict_4_hin = {}
    # links_data["meta"]中每一个key都代表了其对应的关系即（）
    _rows_list = []
    _cols_list = []
    for _r in _meta_keys:
        # 确定当前边类型
        # the key
        _path_type = (_meta[_r][0],_r,_meta[_r][1]) # start_type,path_type,end_type
        
        # 获取对应的关系矩阵
        _mat = links_mat[_r]
        _rows,_cols = _mat.nonzero()
        
        _rows = idx2type_idx(idx_list=_rows, raw_data=raw_data)
        _cols = idx2type_idx(idx_list=_cols, raw_data=raw_data)
        
        _rows_list.append(_rows)
        _cols_list.append(_cols)
        
        
        
        _start = torch.tensor(_rows)
        _end = torch.tensor(_cols)

        
        _data_dict_4_hin[_path_type] = (_start,_end)
    
    _graph = dgl.heterograph(_data_dict_4_hin)
    print(_graph.num_nodes)
    print("------")
    print("nodes_data info:",nodes_data["count"])
    print("links_data info:",links_data["count"])
        """