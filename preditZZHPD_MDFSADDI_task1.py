from numpy.random import seed
import csv
import sqlite3
import time
import numpy as np
import random
import pandas as pd
from pandas import DataFrame
import scipy.sparse as sp
import math
import copy

import deepchem as dc

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import KernelPCA

import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping # 需自加pytorchtools.py
from pytorchtools import BalancedDataParallel# 需自加pytorchtools.py
from radam import RAdam # 需自加radam.py


import torch.nn.functional as F

import networkx as nx

import warnings

warnings.filterwarnings("ignore")

import os
from tensorboardX import SummaryWriter


# 定义随机种子
seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

file_path="./"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_n_heads=4
bert_n_layers=4
drop_out_rating=0.3
batch_size=256
len_after_AE=500
learn_rating=0.00001
epo_num=120
cross_ver_tim=5
cov2KerSize=50
cov1KerSize=25
calssific_loss_weight=5
epoch_changeloss=epo_num//2
weight_decay_rate=0.0001
feature_list = ["smile"]


# 准备药物与药物相互作用的特征和标签
# 这里会产生三个特征矩阵（其中，每一行为一个药物的向量），并获得两个药物拼接在一起的向量之后形成一个矩阵（其中，每一行为两个药物的向量）
def prepare(df_drug, feature_list, mechanism,action,drugA,drugB,ZZHPD_Druglist, ZZHPD_SMILESlist, ZZHPD_drugA, ZZHPD_drugB): # 
    d_label = {}
    d_feature = {}
    # Transfrom the interaction event to number
    # Splice the features
    d_event=[]
    for i in range(len(mechanism)): # len(mechanism) = 37264
        d_event.append(mechanism[i]+" "+action[i]) # ['The risk or severity of adverse effects increase']

    count={}
    for i in d_event: # 计算d_event的频率
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    event_num = len(count)
    list1 = sorted(count.items(), key=lambda x: x[1],reverse=True) 
    # 对d_event的频率进行排序。count.items() 为待排序的对象；key=lambda x: x[1] 为对前面的对象中的第二维数据（即value）的值进行排序，即对事件的多少来排序，reverse=True从大到小进行排序
    
    for i in range(len(list1)): # [['The metabolism decrease', 9810],['The serum concentration increase', 5646]]
        d_label[list1[i][0]]=i # 对The metabolism decrease进行打标签为0，The serum concentration increase打标签为1

    vector = np.zeros((len(np.array(df_drug['name']).tolist())+len(ZZHPD_SMILESlist), 0), dtype=float) # 0矩阵，array([], shape=(572+19, 0), dtype=float64)
    for i in feature_list: # feature_list =['smile']/['target']/['enzyme]
        vector = np.hstack((vector, feature_vector(i, df_drug, ZZHPD_SMILESlist))) #  np.hstack（拼接列）# np.hstack将参数元组的元素数组按水平方向进行叠加 （572，0），（572，572）->(572,572) ;例如：（572，1），（572，572）->(572,573)
   
    for i in range(len(np.array(df_drug['name']).tolist())): # 获取每一种药物的矩阵
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i] # 'Glucosamine' = 特征矩阵的第0行，添加如d_feature字典中

    for i in range(len(np.array(df_drug['name']).tolist()),len(np.array(df_drug['name']).tolist())+len(ZZHPD_SMILESlist)):
        d_feature[ZZHPD_Druglist[i-len(np.array(df_drug['name']).tolist())]] = vector[i] # 'Naringenin' = vector[572]

    new_feature = []
    new_label = []
    ZZHPD_feature = []
    
    for i in range(len(d_event)):
        new_feature.append(np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))) # 拼接两种药物的矩阵
        new_label.append(d_label[d_event[i]])

    for i in range(ZZHPD_drugA.shape[0]): # ZZHPD_drugA.shape[0]= 171 ZZHPD有171种组合
        ZZHPD_feature.append(np.hstack((d_feature[ZZHPD_drugA[i]], d_feature[ZZHPD_drugB[i]])))

    ZZHPD_feature = np.array(ZZHPD_feature)
    new_feature = np.array(new_feature)
    new_label = np.array(new_label)
    return new_feature, new_label, event_num, ZZHPD_feature

## 三个特征矩阵并经过Jaccard和PCA到572维
def feature_vector(feature_name, df, ZZHPD_SMILESlist):
    # df are the 572 kinds of drugs
    # Jaccard Similarity
    def Jaccard(matrix): # 雅卡尔系数能够量度有限样本集合的相似度，其定义为两个集合交集大小与并集大小之间的比例
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator

    # 创建特征矩阵和进行赋值
    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]

    # 确定特征矩阵有多少行和列，以及列名，全0表
    for i in drug_list:
        for each_feature in i.split('|'): 
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # all_feature = 583

    # ZZHPD
    for i in ZZHPD_SMILESlist:
        for each_ZZHPD in i.split('|'):
            if each_ZZHPD not in all_feature:
                all_feature.append(each_ZZHPD) # 595,加了12个特征

    feature_matrix = np.zeros((len(drug_list)+len(ZZHPD_SMILESlist), len(all_feature)), dtype=float) # df[smiles]_shape:(591,595)=(572+19,583+12)，df[target]_shape:(572,1162)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # 构成特征矩阵，列名为各不同的特征

    # 为特征矩阵空表填1
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'): # 为特征矩阵进行赋值1
            df_feature[each_feature].iloc[i] = 1  #  df_feature[9].iloc[0]=1 # 特征为9的列+第0行进行赋值
    
    # ZZHPD表填1
    for i in range(len(drug_list),len(drug_list)+len(ZZHPD_SMILESlist)):
        for each_ZZHPD in ZZHPD_SMILESlist[i-len(drug_list)].split('|'): # ZZHPD_SMILESlist[0]对应的是df.iloc[572]
            df_feature[each_ZZHPD].iloc[i] = 1  
    
    df_feature = np.array(df_feature) # df[smiles]的shape:(591,595)通过雅卡尔系数变成(591,591)
    sim_matrix = np.array(Jaccard(df_feature))

    print(feature_name + " len is:" + str(len(sim_matrix[0]))) # 591中的第0个矩阵sim_matrix[0]

    return sim_matrix


class DDIDataset(Dataset):
    def __init__(self, x, y):
        self.len = len(x)
        self.x_data = torch.from_numpy(x) # 用来将数组array转换为张量Tensor

        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len




class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)
        X = self.AN1(output + X)

        output = self.l1(X)
        X = self.AN2(output + X)

        return X




def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))



class AE1(torch.nn.Module):  # Joining together
    def __init__(self, vector_size):
        super(AE1, self).__init__()

        self.vector_size = vector_size

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.att2 = EncoderLayer((self.vector_size + len_after_AE) // 2, bert_n_heads)
        self.l2 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, len_after_AE)

        self.l3 = torch.nn.Linear(len_after_AE, (self.vector_size + len_after_AE) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.att2(X)
        X = self.l2(X)

        X_AE = self.dr(self.bn3(self.ac(self.l3(X))))

        X_AE = self.l4(X_AE)

        return X, X_AE




class AE2(torch.nn.Module):  # twin network
    def __init__(self, vector_size):
        super(AE2, self).__init__()

        self.vector_size = vector_size // 2

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE // 2) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE // 2) // 2)

        self.att2 = EncoderLayer((self.vector_size + len_after_AE // 2) // 2, bert_n_heads)
        self.l2 = torch.nn.Linear((self.vector_size + len_after_AE // 2) // 2, len_after_AE // 2)

        self.l3 = torch.nn.Linear(len_after_AE // 2, (self.vector_size + len_after_AE // 2) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE // 2) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + len_after_AE // 2) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size]
        X2 = X[:, self.vector_size:]

        X1 = self.dr(self.bn1(self.ac(self.l1(X1))))
        X1 = self.att2(X1)
        X1 = self.l2(X1)
        X_AE1 = self.dr(self.bn3(self.ac(self.l3(X1))))
        X_AE1 = self.l4(X_AE1)

        X2 = self.dr(self.bn1(self.ac(self.l1(X2))))
        X2 = self.att2(X2)
        X2 = self.l2(X2)
        X_AE2 = self.dr(self.bn3(self.ac(self.l3(X2))))
        X_AE2 = self.l4(X_AE2)

        X = torch.cat((X1, X2), 1)
        X_AE = torch.cat((X_AE1, X_AE2), 1)

        return X, X_AE





class cov(torch.nn.Module):
    def __init__(self, vector_size):
        super(cov, self).__init__()

        self.vector_size = vector_size

        self.co2_1 = torch.nn.Conv2d(1, 1, kernel_size=(2, cov2KerSize))
        self.co1_1 = torch.nn.Conv1d(1, 1, kernel_size=cov1KerSize)
        self.pool1 = torch.nn.AdaptiveAvgPool1d(len_after_AE)

        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size // 2]
        X2 = X[:, self.vector_size // 2:]

        X = torch.cat((X1, X2), 0)

        X = X.view(-1, 1, 2, self.vector_size // 2)

        X = self.ac(self.co2_1(X))

        X = X.view(-1, self.vector_size // 2 - cov2KerSize + 1, 1)
        X = X.permute(0, 2, 1)
        X = self.ac(self.co1_1(X))

        X = self.pool1(X)

        X = X.contiguous().view(-1, len_after_AE)

        return X




class ADDAE(torch.nn.Module):
    def __init__(self, vector_size):
        super(ADDAE, self).__init__()

        self.vector_size = vector_size // 2

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + len_after_AE) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.att1 = EncoderLayer((self.vector_size + len_after_AE) // 2, bert_n_heads)
        self.l2 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, len_after_AE)
        # self.att2=EncoderLayer(len_after_AE//2,bert_n_heads)

        self.l3 = torch.nn.Linear(len_after_AE, (self.vector_size + len_after_AE) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + len_after_AE) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + len_after_AE) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size]
        X2 = X[:, self.vector_size:]
        X = X1 + X2

        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.att1(X)
        X = self.l2(X)

        X_AE = self.dr(self.bn3(self.ac(self.l3(X))))

        X_AE = self.l4(X_AE)
        X_AE = torch.cat((X_AE, X_AE), 1)

        return X, X_AE




class BERT(torch.nn.Module):
    def __init__(self, input_dim, n_heads, n_layers, event_num):
        super(BERT, self).__init__()

        self.ae1 = AE1(input_dim)  # Joining together
        self.ae2 = AE2(input_dim)  # twin loss
        self.cov = cov(input_dim)  # cov
        self.ADDAE = ADDAE(input_dim)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.input_dim = input_dim

        self.layers = torch.nn.ModuleList([EncoderLayer(len_after_AE * 5, n_heads) for _ in range(n_layers)])
        self.AN = torch.nn.LayerNorm(len_after_AE * 5)

        self.l1 = torch.nn.Linear(len_after_AE * 5, (len_after_AE * 5 + event_num) // 2)
        self.bn1 = torch.nn.BatchNorm1d((len_after_AE * 5 + event_num) // 2)

        self.l2 = torch.nn.Linear((len_after_AE * 5 + event_num) // 2, event_num)

        self.ac = gelu

    def forward(self, X):
        X1, X_AE1 = self.ae1(X)
        X2, X_AE2 = self.ae2(X)

        X3 = self.cov(X)

        X4, X_AE4 = self.ADDAE(X)

        X5 = X1 + X2 + X3 + X4

        X = torch.cat((X1, X2, X3, X4, X5), 1)

        for layer in self.layers:
            X = layer(X)
        X = self.AN(X)

        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.l2(X)

        return X, X_AE1, X_AE2, X_AE4




class focal_loss(nn.Module):
    def __init__(self, gamma=2):
        super(focal_loss, self).__init__()

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim() == 2 and labels.dim()==1
        labels = labels.view(-1, 1)  # [B * S, 1]
        preds = preds.view(-1, preds.size(-1))  # [B * S, C]

        preds_logsoft = F.log_softmax(preds, dim=1)  # 先softmax, 然后取log
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels)  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels)

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = loss.mean()

        return loss


class my_loss1(nn.Module):
    def __init__(self):
        super(my_loss1, self).__init__()

        self.criteria1 = torch.nn.CrossEntropyLoss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, X, target, inputs, X_AE1, X_AE2, X_AE4):
        loss = calssific_loss_weight * self.criteria1(X, target) + \
               self.criteria2(inputs.float(), X_AE1) + \
               self.criteria2(inputs.float(), X_AE2) + \
               self.criteria2(inputs.float(), X_AE4)
        return loss


class my_loss2(nn.Module):
    def __init__(self):
        super(my_loss2, self).__init__()

        self.criteria1 = focal_loss()
        self.criteria2 = torch.nn.MSELoss()

    def forward(self, X, target, inputs, X_AE1, X_AE2, X_AE4):
        loss = calssific_loss_weight * self.criteria1(X, target) + \
               self.criteria2(inputs.float(), X_AE1) + \
               self.criteria2(inputs.float(), X_AE2) + \
               self.criteria2(inputs.float(), X_AE4)
        return loss



def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, alpha)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y



def BERT_train(model, x_train, y_train, x_test, y_test, event_num):
    model_optimizer = RAdam(model.parameters(), lr=learn_rating, weight_decay=weight_decay_rate)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    x_train = np.vstack((x_train, np.hstack((x_train[:, len(x_train[0]) // 2:], x_train[:, :len(x_train[0]) // 2])))) # 沿着竖直方向将矩阵堆叠起来，len(x_train[0]) // 2:后面1716列调到前面来，即drugAdrugB变成drugBdrugA
    y_train = np.hstack((y_train, y_train))
    np.random.seed(seed)
    np.random.shuffle(x_train) # 打乱每一行
    np.random.seed(seed)
    np.random.shuffle(y_train)

    len_train = len(y_train)
    len_test = len(y_test)
    print("arg train len", len(y_train))
    print("test len", len(y_test))

    train_dataset = DDIDataset(x_train, np.array(y_train))
    test_dataset = DDIDataset(x_test, np.array(y_test))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # batch_size=256
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epo_num): # epo_num = 120
        if epoch < epoch_changeloss: # epoch_changeloss = 60
            my_loss = my_loss1()
        else:
            my_loss = my_loss2()

        running_loss = 0.0

        model.train()
        for batch_idx, data in enumerate(train_loader, 0): # 指定batch_idx的索引从0开始
            x, y = data

            lam = np.random.beta(0.5, 0.5)
            index = torch.randperm(x.size()[0]).cuda()
            inputs = lam * x + (1 - lam) * x[index, :]

            targets_a, targets_b = y, y[index]

            inputs = inputs.to(device)
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)

            model_optimizer.zero_grad()
            # forward + backward+update
            X, X_AE1, X_AE2, X_AE4 = model(inputs.float())

            loss = lam * my_loss(X, targets_a, inputs, X_AE1, X_AE2, X_AE4) + (1 - lam) * my_loss(X, targets_b, inputs,
                                                                                                  X_AE1, X_AE2, X_AE4)

            loss.backward()
            model_optimizer.step()
            running_loss += loss.item()

        model.eval()
        testing_loss = 0.0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader, 0):
                inputs, target = data

                inputs = inputs.to(device)

                target = target.to(device)

                X, X_AE1, X_AE2, X_AE4 = model(inputs.float())

                loss = my_loss(X, target, inputs, X_AE1, X_AE2, X_AE4)
                testing_loss += loss.item()
        print('epoch [%d] loss: %.6f testing_loss: %.6f ' % (
        epoch + 1, running_loss / len_train, testing_loss / len_test))

    pre_score = np.zeros((0, event_num), dtype=float)
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            inputs, _ = data
            inputs = inputs.to(device)
            X, _, _, _ = model(inputs.float())
            pre_score = np.vstack((pre_score, F.softmax(X).cpu().numpy()))
    return pre_score



def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)



def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                         average=None)
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]



def cross_val(feature,label,event_num):
    skf = StratifiedKFold(n_splits=cross_ver_tim) # shuffle：默认为False；# random_state：默认为None，表示随机数的种子，只有当shuffle设置为True的时候才会生效。
    y_true = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    y_pred = np.array([])
    
    for train_index, test_index in skf.split(feature, label):
        
        model=BERT(len(feature[0]),bert_n_heads,bert_n_layers,event_num) # len(feature[0])=3432

        X_train, X_test = feature[train_index], feature[test_index]
        y_train, y_test = label[train_index], label[test_index]
        print("train len", len(y_train))
        print("test len", len(y_test))
        
        pred_score=BERT_train(model,X_train,y_train,X_test,y_test,event_num)
        
        pred_type = np.argmax(pred_score, axis=1)
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))

        y_true = np.hstack((y_true, y_test))
        
    result_all, result_eve= evaluate(y_pred, y_score, y_true, event_num)

    return result_all, result_eve



def save_result(filepath,result_type,result):
    with open('ZZHPD_dataset1_smile' + filepath+result_type +'task1_MDF_SA_DDI'+ '.csv', "w", newline='',encoding='utf-8') as csvfile: # 
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0



def ZZHPD(drugfile, smilesfile,drugABfile):
    # ZZHPD_druglist
    ZZHPD_druglist = []
    with open(drugfile,'r') as f:
        for line in f:
            ZZHPD_druglist.append(line.splitlines()[0])

    # ZZHPD_smiles
    df_ZZHPD = pd.read_csv(smilesfile)
    smiles = df_ZZHPD['canonical_smiles'].tolist()
    # ZZHPD的smiles进行编码
    featurizer = dc.feat.PubChemFingerprint()
    pubchem = np.zeros((0, 881), dtype=float)
    for i in smiles:
        pubchemcoding = np.squeeze(featurizer.featurize(i)) # np.squeeze() : 去掉numpy数组冗余的维度
        pubchem = np.row_stack((pubchem,pubchemcoding)) # 行拼接

    df_ZZHPD_pubchem = pd.DataFrame(pubchem)

    ZZHPD_smileslist = []
    for i in range(df_ZZHPD_pubchem.shape[0]): # 先行后列
        s = ''
        for j in range(df_ZZHPD_pubchem.shape[1]):
            if df_ZZHPD_pubchem.iat[i,j] == 1: # df_ZZHPD_pubchem.iat[i,j]，第i行第j列
                s = s + str(j) + '|'
        s = s[:-1] # 不要最后一个|
        ZZHPD_smileslist.append(s)
  
    ZZHPD_extraction = pd.read_csv(drugABfile)
    ZZHPD_drugA = ZZHPD_extraction['drugA']
    ZZHPD_drugB = ZZHPD_extraction['drugB']
    return ZZHPD_druglist, ZZHPD_smileslist, ZZHPD_drugA, ZZHPD_drugB



def BERT_test(model, x_train, y_train, x_test, event_num):
    model_optimizer = RAdam(model.parameters(), lr=learn_rating, weight_decay=weight_decay_rate)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    x_train = np.vstack((x_train, np.hstack((x_train[:, len(x_train[0]) // 2:], x_train[:, :len(x_train[0]) // 2])))) # 沿着竖直方向将矩阵堆叠起来，len(x_train[0]) // 2:后面1716列调到前面来，即drugAdrugB变成drugBdrugA
    y_train = np.hstack((y_train, y_train))
    np.random.seed(seed)
    np.random.shuffle(x_train) # 打乱每一行
    np.random.seed(seed)
    np.random.shuffle(y_train)

    len_train = len(y_train)

    print("arg train len", len(y_train))

    train_dataset = DDIDataset(x_train, np.array(y_train))
    # test_dataset = DDIDataset(x_test, np.array(y_test))
    test_dataset = torch.from_numpy(x_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # batch_size=256
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epo_num): # epo_num = 120
        if epoch < epoch_changeloss: # epoch_changeloss = 60
            my_loss = my_loss1()
        else:
            my_loss = my_loss2()

        running_loss = 0.0

        model.train()
        for batch_idx, data in enumerate(train_loader, 0): # 指定batch_idx的索引从0开始
            x, y = data

            lam = np.random.beta(0.5, 0.5)
            index = torch.randperm(x.size()[0]).cuda()
            inputs = lam * x + (1 - lam) * x[index, :]

            targets_a, targets_b = y, y[index]

            inputs = inputs.to(device)
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)

            model_optimizer.zero_grad()
            # forward + backward+update
            X, X_AE1, X_AE2, X_AE4 = model(inputs.float())

            loss = lam * my_loss(X, targets_a, inputs, X_AE1, X_AE2, X_AE4) + (1 - lam) * my_loss(X, targets_b, inputs,
                                                                                                  X_AE1, X_AE2, X_AE4)

            loss.backward()
            model_optimizer.step()
            running_loss += loss.item()
        print('epoch [%d] loss: %.6f' % (epoch + 1, running_loss / len_train))

    pre_score = np.zeros((0, event_num), dtype=float)
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            inputs = data
            inputs = inputs.to(device)
            X, _, _, _ = model(inputs.float())
            pre_score = np.vstack((pre_score, F.softmax(X).cpu().numpy()))
    return pre_score



def predict_ZZHPD(feature, label, ZZHPD_feature, event_num):
        
    model = BERT(len(feature[0]),bert_n_heads,bert_n_layers,event_num) # len(feature[0])=3432

    pred_score = BERT_test(model, feature, label, ZZHPD_feature,event_num)
    
    pred_type = np.argmax(pred_score, axis=1)

    return pred_type



def main():
    
    conn = sqlite3.connect("./event.db")
    
    df_drug = pd.read_sql('select * from drug;', conn)
    extraction = pd.read_sql('select * from extraction;', conn)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']
    
    ZZHPD_Druglist, ZZHPD_SMILESlist, ZZHPD_drugA, ZZHPD_drugB = ZZHPD('ZZHPD_drug.txt','ZZHPD_smlies.csv','ZZHPD_DrugAB.csv')
    
    new_feature, new_label, event_num, ZZHPD_feature = prepare(df_drug, feature_list, mechanism,action,drugA,drugB,ZZHPD_Druglist, ZZHPD_SMILESlist, ZZHPD_drugA, ZZHPD_drugB) # 用到event中的drug表和extraction表
    
    np.random.seed(seed)
    np.random.shuffle(new_feature)
    np.random.seed(seed)
    np.random.shuffle(new_label)
    print("dataset len", len(new_feature))
    
    start=time.time()
    result_all, result_eve=cross_val(new_feature,new_label,event_num)
    print("time used:", (time.time() - start) / 3600)
    save_result(file_path,"all",result_all)
    save_result(file_path,"each",result_eve)

    predict_ZZHPDlabel = predict_ZZHPD(new_feature, new_label, ZZHPD_feature, event_num)
    df_predict_ZZHPDlabel = pd.DataFrame(predict_ZZHPDlabel).to_csv("./predictMDFSADDI_ZZHPDlabel.csv", encoding='utf-8') 




main()


