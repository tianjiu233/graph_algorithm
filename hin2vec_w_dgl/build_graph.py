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



# ------ 构建异构图 ------
def idx2type_idx(idx_list,raw_data):
    shift_dict = raw_data.nodes["shift"]
    def map_idx(idx):
        _type = raw_data.get_node_type(idx)
        idx2type = idx - shift_dict[_type]
        return idx2type
    new_idx_list = list(map(map_idx,idx_list))
    return new_idx_list


def build_hete_graph(data_dir):
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
    _data_dict_4_hete = {}
    # links_data["meta"]中每一个key都代表了其对应的关系即（）
    for _r in _meta_keys:
        # 确定当前边类型
        _path_type = (_meta[_r][0],_r,_meta[_r][1]) # start_type,path_type,end_type
        # 获取对应的关系矩阵
        _mat = links_mat[_r]
        _rows,_cols = _mat.nonzero()
        
        _rows = idx2type_idx(idx_list=_rows, raw_data=raw_data)
        _cols = idx2type_idx(idx_list=_cols, raw_data=raw_data)
        
        _s = torch.tensor(_rows)
        _e = torch.tensor(_cols)
        
        _data_dict_4_hete[_path_type] = (_s,_e)
    
    _graph = dgl.heterograph(_data_dict_4_hete)
    return _graph,raw_data



def generate_metapath(link_len,edge_type_dict):
    
    _path_dict = edge_type_dict
    meta_path_type = list(_path_dict.keys())
    
    _path_sample = []
    _cur_path_list = meta_path_type
    for i in range(link_len):
        _p = random.choice(_cur_path_list)
        _path_sample.append(_p)
        # update _cur_path_list
        _cur_path_list = _path_dict[_p]
    
    return _path_sample

def generate_metapath_w_assigned_start(link_len,edge_type_dict,start_link_type=0):
    
    _path_dict = edge_type_dict
    
    _path_sample = []
    # 起始点指定产生特定的_path_sample
    _p = start_link_type
    _path_sample.append(_p)
    _cur_path_list = _path_dict[_p]
    
    for i in range(1,link_len):
        _p = random.choice(_cur_path_list)
        _path_sample.append(_p)
        # update _cur_path_list
        _cur_path_list = _path_dict[_p]
    
    return _path_sample


if __name__=="__main__":
    
    # 
    data_name = "DBLP"
    data_dir = f'./nc_data/{data_name}'
    
    hete_graph,raw_data = build_hete_graph(data_dir)
    # home_graph,raw_data = build_homo_graph(data_dir)
    
    
    links_type = raw_data.load_links()["meta"]
    """
    1-hop
    {0: (0, 1), 3: (1, 0), 1: (1, 2), 2: (1, 3), 4: (2, 1), 5: (3, 1)}
    
    除了这样的1-hop连接之外还要考虑到多hop,如2-hop
    
    0-1-2: 0,1
    0-1-3: 0,2
    
    1-0-1: 3,0
    1-2-1: 1,4
    1-3-1: 2,5
    
    2-1-0: 4,3
    2-1-2: 4,1
    2-1-3: 4,2
    
    3-1-0: 5,3
    3-1-2: 5,1
    3-1-3: 5,2
    
    但是这种生成方式太过复杂了，一种相对简单的生成方式，则是通过随机生成
    
    比如生成的逻辑字典：
    s:e_list
    
    _path_dict = {
        0:[1,2],
        3:[0],
        1:[4]
        2:[5]
        4:[1,2,3]
        5:[1,2,3]
        }
    
    """
    edge_path_dict = {
        0:[1,2],
        3:[0],
        1:[4],
        2:[5],
        4:[1,2,3],
        5:[1,2,3],
        }
    
    
    meta_path_sample = generate_metapath(4,edge_path_dict)
    # meta_path_sample = [0,2,5,3]
    
    # 在_graph的基础上进行随机游走
    # _: 对应的每个节点的类型
    traces,_ = dgl.sampling.random_walk(g = hete_graph, 
                                        nodes = [0,0,0,0],
                                        metapath=meta_path_sample,)
    
    meta_path_sample_w_start = generate_metapath_w_assigned_start(4, edge_path_dict,3)
    traces,_ = dgl.sampling.random_walk(g = hete_graph, 
                                        nodes = [0,0,0,0],
                                        metapath=meta_path_sample_w_start,)
