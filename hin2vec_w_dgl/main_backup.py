# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 08:05:55 2023

@author: huijian
"""

import torch
import dgl
import random
import numpy as np

# mylib
from build_graph import build_hete_graph
from build_graph import generate_metapath

if __name__=="__main__":
    
    print("Testing hin2vec")
    
    # 相关参数设置，rw_walk_length,相关元路径的类型等
    w_hop = 2
    
    """ DBLP 相关的参数
    对于dblp数据集而言，edge_type_dict代表了不同链接了哦下哦美好(link)之后可能出现的链接类型
    基础的1-hop元路径一共有5种，我们可以尝试计算由其延伸出来的元路径一共多少种
    
    """
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
    
    def find_all_w_hop(w_hop,edge_path_dict,flag="-"):
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
    
    relations = []
    for i in range(1,w_hop+1):
        tmp = find_all_w_hop(w_hop=i,edge_path_dict=edge_path_dict,flag="-")
        relations = relations + tmp

    # the metapath within 2-hop could be 
    # ['0', '3', '1', '2', '4', '5', '0-1', '0-2', '3-0', '1-4', '2-5', '4-1', '4-2', '4-3', '5-1', '5-2', '5-3']
    # 根据relation 就可以确定path_size
    # 理论上需要构建字典确认各种metapath与其之间的关系
    path_idx2relation_dict = {}
    relation2path_idx_dict = {}
    idx_key = 0
    for _r in relations:
        path_idx2relation_dict[idx_key] = _r
        relation2path_idx_dict[_r] = idx_key
        idx_key = idx_key + 1
    
    
    # 2.prepare the data
    # 构建图
    hete_graph,raw_data = build_hete_graph(data_dir)
    node_size = raw_data.nodes["total"]
    path_size = len(relations)
        
    
    # 完成图构建之后进行随机游走采样[实验代码]
    metapath_length = 64
    metapath = generate_metapath(metapath_length,edge_path_dict)
    # traces,sampled_nodes_types = dgl.sampling.random_walk(g = hete_graph, nodes = [0,0,0,0],metapath=metapath,)
    # metapath_length的长度为 5+1，涉及6个节点，考虑的w_hop = 2
    # 因此，对于每个节点来说，考虑的关系为对于i节点,考虑的metapath为metapath[i]...metapath[i+w_hop-1]构成的
    # 因此对于前l-w_hop个节点，每个节点能够具有w_hop个关系，因此有(l-w_hop)*w_hop个样本[来自于每次随机游走]
    # 每当metapath已经生成时，我们就可以生成对应的标签，其前l-w_hop个节点的标签
    sampled_relations_list = []
    flag = "-"
    for sampled_node_idx in range(metapath_length+1 - w_hop):
        _bias = sampled_node_idx
        for i in range(w_hop):
            if i == 0:
                _r = metapath[_bias]
                _r = str(_r)
            else:
                _r = _r + flag + str(metapath[_bias+i])
            sampled_relations_list.append(_r)
    # 将sampled_relations_list映射到sampled_path_idx_list
    def _map(_r):
        path_idx_list = relation2path_idx_dict[_r]
        return path_idx_list
    sampled_path_idx_list = list(map(_map,sampled_relations_list))
    
    
    # 对于使用dgl进行采样，需要明确其最大的nodes的范围，这一点需要根据其metapath的第一个元素标签确定
    node_type = raw_data.load_links()["meta"][metapath[0]][0]
    _typed_node_count = raw_data.nodes["count"][node_type]
    
    batch_size = 32
    # start_nodes = random.sample(range(0,_typed_node_count),batch_size)
    start_nodes = [random.randint(0,_typed_node_count-1) for _ in range(batch_size)] # list
    # traces: torch.int32 (len(start_nodes,metapath_length+1))
    traces,sampled_nodes_types = dgl.sampling.random_walk(g = hete_graph, nodes = start_nodes,metapath=metapath,)
    
    """
    # 随后就是构建3元组形成训练数据集，同时也要考虑traces中存在-1的问题
    ## 如果第n个点为-1[n一定大于0.因为起始点一定存在]，则第 (n-1)*2,(n-1)*2+1个label或者说样例就都不存在了，以n为2为例，则2，3以及以后的label失效
    # 但是这样是针对个体进行的，如何使得其能使得batch中的多个样本直接生成
    # 将traces与sampled_nodes_types 结合起来，最终应该有batch_size*(length-w_hop)*w_hop个样本
    # 将trace先转为np.array方便进行操作
    train_data = []
    traces = traces.cpu().detach().numpy() # np.array
    sampled_nodes_types = sampled_nodes_types.cpu().detach().numpy() # np.array
    _shift_dict = raw_data.nodes["shift"]
    for _t in traces:
        # process every traces
        for i in range(metapath_length+1-w_hop):
            typed_start_node_idx = _t[i]
            # 在start_node_idx处进行break是没有意义的
            # if typed_start_node_idx < 0:
              #  break
            start_node_type = sampled_nodes_types[i]
            # global idx
            start_node_idx = typed_start_node_idx + _shift_dict[start_node_type]
            for j in range(1,w_hop+1): # 1,2,w_hop
                typed_end_node_idx = _t[i+j]
                if typed_end_node_idx < 0:
                    break
                end_node_type = sampled_nodes_types[i+j]
                # global idx
                end_node_idx = typed_end_node_idx + _shift_dict[end_node_type]
                _path_idx = sampled_path_idx_list[i*w_hop + j - 1]
                train_sample = [start_node_idx,end_node_idx,_path_idx]     
                train_data.append(train_sample)
            
        
    """
    
    # 3.build the model
    # 构建模型，这里构建模型可以分为多种，一种是nn.module本身，还有一种则是将随机采样和图数据共同构成模型，以避免使用过大的内存保存图数据
    
    # 4.获取嵌入
    
    # 5.使用svm或lr等方法进行改进