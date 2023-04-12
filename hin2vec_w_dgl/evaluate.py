# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:52:15 2023

@author: bayes
"""

import re
import numpy as np
import os
from collections import defaultdict
from itertools import *
import argparse
import json
import pickle
import sklearn
from sklearn import linear_model
import sklearn.metrics as Metric
import sys

class EvaluateEmbeds():
    def __init__(self,dl_pickle='',embed_f='',model='others',dataset='DBLP',data_root='./data/NC/',need_transf=True):
        """
        初始化，输入dl的pickle地址和embed_f的地址。
        标准形式的embeds 为 np.array 2d， shape是(node_n,embed_d)
        若为其他模型，默认embeds存储为dict，即dict(str(node_id):embed),默认model为others，调用dict2array转换为标准形式。
        若模型为HetGNN，embeds存储为txt，调用get_embed_hetgnn，读取并转换为标准形式
        若模型为GATNE，embeds按关系存储为字典形式，在初始化函数中转换为标准形式
        特别地，GATNE的嵌入有用于lp的和用于nc的两种形式，
            用于lp的：每种关系给一个嵌入2d array
            用于nc的：将节点所有关系下的拼接作为节点最终嵌入
        :param dl_pickle: dataloader存储地址
        :param embed_f: 嵌入文件存储地址
        :param model: 模型名，针对HetGNN和GATNE定义特别的读取形式
        :param dataset: 默认数据集为DBLP
        :param data_root:数据集所在地址的父目录
        :param need_transf: 读取到的嵌入是否需要再转换一下
        """
        self.model = model
        # 数据信息
        info_path = os.path.join(data_root,f"{dataset}/info.dat")
        self.data_info = json.load(open(info_path, 'r'))

        # 获取 dl
        print(dl_pickle)
        if dl_pickle_f != '' and os.path.exists(dl_pickle_f) :
            self.dl = pickle.load(open(dl_pickle_f, 'rb'))
            print(f'Info: load {dataset} from {dl_pickle_f}')
        else:
            self.dl = data_loader(os.path.join(root_dir, f"data/NC/{dataset}"))
            # pickle.dump(self.dl, open(dl_pickle_f, 'wb'),protocol=3)
            print(f'Info: load {dataset} from original data')

        node_n = self.dl.nodes['total']
        node_shift = self.dl.nodes['shift']

        # 获取embeds



    def do_nc(self):
        """
        节点分类评估
        :param node_embeds: np.array 2d [node_n,embed_d]
        :param dl: data_loader
        """
        "获取训练集嵌入和标签"
        train_id = np.where(self.dl.labels_train['mask'])
        train_features = self.embeds[train_id]
        train_target = self.dl.labels_train['data'][train_id]
        train_target = [np.argmax(l)for l in train_target] # 从one-hot获取label
        train_target = np.array(train_target) # [train_n,]

        "训练逻辑回归模型"
        learner = linear_model.LogisticRegression()
        learner.fit(train_features, train_target)
        print("training finish!")

        "获取测试集嵌入，完成预测，获取测试集label用于评估"
        test_id = np.where(self.dl.labels_test['mask'])
        test_features = self.embeds[test_id]
        test_target = self.dl.labels_test['data'][test_id]
        test_target = [np.argmax(l) for l in test_target]
        test_target = np.array(test_target)

        test_predict = learner.predict(test_features)
        print("test prediction finish!")

        "评估指标"
        print ("MicroF1: ")
        print (sklearn.metrics.f1_score(test_target,test_predict,average='micro'))
        print("MacroF1: ")
        print(sklearn.metrics.f1_score(test_target, test_predict, average='macro'))






