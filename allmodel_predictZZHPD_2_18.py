from cmath import log
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools

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

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import KernelPCA
# from sklearn.linear_model.logistic import LogisticRegression # 版本0.21.2
from sklearn.linear_model import LogisticRegression # 版本1.0.2


import tensorflow as tf
from tensorflow.keras.models import Model  #
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, BatchNormalization #
from tensorflow.keras.callbacks import EarlyStopping #

import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
# from pytorchtools import EarlyStopping # MDFSADDI没用EarlyStopping，不注释这句会覆盖from tensorflow.keras.callbacks import EarlyStopping
# from pytorchtools import BalancedDataParallel
from radam import RAdam
import torch.nn.functional as F

import networkx as nx

import warnings
warnings.filterwarnings("ignore")

import os

import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

# 数据不平衡处理
from imblearn.over_sampling import SMOTE # 少类：过抽样(上采样)——复制少数类的样本/加入随机噪声、干扰数据
from collections import Counter

from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler

# 固定pytorch/tensorflow神经网络中每次初始化的参数，得到同样的预测结果(对ZZHPD的drugpair的预测每次都一样)
seed = 0
random.seed(seed) # python的随机性
os.environ['PYTHONHASHSEED'] = str(seed) # 设置python哈希种子，为了禁止hash随机化
np.random.seed(seed) # np的随机性

torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed) # 若使用多个GPU，为所有的GPU设置种子
torch.backends.cudnn.deterministic = True # 选择确定性算法
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1' 
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from tensorflow.keras.initializers import glorot_normal
# dense = Dense(kernel_initializer=glorot_normal(seed=seed))
# drop = Dropout(seed=seed)
# from tfdeterminism import patch # tfdeterminism 暂时只适用于小于2.1的tensorflow版本，因为目前没有适用于TensorFlow 2.1版或更高版本的修补程序
# patch()

file_path="./"


repeat_running = 3 # 重复运行的次数为10

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

droprate = 0.3

all_eval_type = 10 # 二分类为10(加4个，分别为TN,FP,FN,TP)，多分类为11 
each_eval_type = 6

# log_dir = './tmp'+ str(time.time()) + "/"
log_dir = './logging/' + time.strftime('%Y_%m_%d',time.localtime(time.time()))+ "/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def logging(msg):
  fpath = os.path.join(log_dir,"log.txt") # './tmp1647661797.9780576/log.txt'
  with open(fpath, "a" ) as fw: # a:打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
    fw.write("%s\n" % msg)
  print(msg)

def prepare(df_drug, feature_list, mechanism, action, drugA, drugB,clfName,ZZHPD_Druglist, ZZHPD_SMILESlist, ZZHPD_drugA, ZZHPD_drugB, antidp_val):
    d_label = {}
    d_feature = {}

    # Transfrom the interaction event to number
    d_event = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])

    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    event_num = len(count)
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    # pd.DataFrame(list1).to_csv("./dataset2_label_frequency.csv", encoding='utf-8')
    # pd.DataFrame(list1).to_csv("./dataset1_label_frequency.csv", encoding='utf-8')
    for i in range(len(list1)):
        d_label[list1[i][0]] = i # {'The metabolism decrease': 0}
    
    vector = np.zeros((len(np.array(df_drug['name']).tolist())+len(ZZHPD_SMILESlist), 0), dtype='float32')  
    # shape=(1276, 0) 1258+18  # 改成float32, 之前是float64
    # shape=(1278, 0) 1258+20  # 改成float32, 之前是float64 # 原数据集里有1258种药物+ZZHPD的20个药物= 1278
    for i in feature_list: # feature_list为smiles
        tempvec = feature_vector(i, df_drug, clfName, ZZHPD_SMILESlist) ###
        vector = np.hstack((vector, tempvec)) 
    # Transfrom the drug ID to feature vector
    for i in range(len(np.array(df_drug['name']).tolist())): # 每个药物一个特征向量，共1258个药物，每个药物有1278维
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]

    #ZZHPD_druglist ###
    for i in range(len(np.array(df_drug['name']).tolist()),len(np.array(df_drug['name']).tolist())+len(ZZHPD_SMILESlist)):
        d_feature[ZZHPD_Druglist[i-len(np.array(df_drug['name']).tolist())]] = vector[i] # 'Naringenin' = vector[572]    # 2048维的ECFP4编码， 每个药物的长度为1276

    # Use the dictionary to obtain feature vector and label
    new_feature = []
    new_label = []
    ZZHPD_feature = [] ###

    antidp_feature = []
    antidp_label = []

    # df_com_antidp = pd.read_csv('common_antidp_19.csv')
    df_com_antidp = pd.read_csv('DDIMDL/common_antidp_42_drugbankdoc.csv')
    list_com_antidp = [antidp for antidp in df_com_antidp['Antidepressant']]
    df_com_antidp_mechanism = pd.DataFrame(columns=('index','drugA','drugB','event','label')) # 创建一个空DataFrame,并命名列名
    df_all_mechanism = pd.DataFrame(columns=('index','drugA','drugB','event','label')) # 创建一个空DataFrame,并命名列名
    j = 0
    if antidp_val == 'Yes':
        
        for i in range(len(d_event)):
            if drugA[i] in list_com_antidp or drugB[i] in list_com_antidp: # 有列表中的抗抑郁的药物关系都要进这里，目的是把把列表中的抗抑郁药物当成未知药物
                # j = j + 1 # 1693
                if drugA[i] in list_com_antidp and drugB[i] in list_com_antidp: # 只记录抗抑郁药物之间的关系，抗抑郁药物与其他药物的关系不考虑
                    j = j + 1 # 统计抗抑郁药物之间关系的数量
                    temp_antidp = np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))
                    antidp_feature.append(temp_antidp)
                    antidp_label.append(d_label[d_event[i]])

                    row_com_antidp_mechanism = {'index':i,'drugA':drugA[i],'drugB':drugB[i],'event':d_event[i],'label':d_label[d_event[i]]}
                    df_com_antidp_mechanism.loc[j] = row_com_antidp_mechanism

            else:
                temp = np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))
                new_feature.append(temp)
                new_label.append(d_label[d_event[i]])
        df_com_antidp_mechanism.reset_index(drop=True).to_csv('df_com_antidp_mechanism' +  '.csv', encoding='utf-8') # drop=True不保留之前索引，重置索引之后才输出csv文件
        logging("The mechanism of antidepressant numbered %s" %(j)) # 抗抑郁药物的mechanism一共有多少对
        logging("The mechanism of training numbered %s" %(len(new_feature)))
    else:
        for i in range(len(d_event)):
            j = j + 1
            temp = np.hstack((d_feature[drugA[i]], d_feature[drugB[i]])) # 药物A和药物B相加拼接
            new_feature.append(temp)
            new_label.append(d_label[d_event[i]])

            # row_all_mechanism = {'index':i,'drugA':drugA[i],'drugB':drugB[i],'event':d_event[i],'label':d_label[d_event[i]]}
            # df_all_mechanism.loc[j] = row_all_mechanism # 第0行为列名，第一行开始存储内容
        # df_all_mechanism.to_csv('df_all_mechanism' +  '.csv', encoding='utf-8') # drop=True不保留之前索引，重置索引之后才输出csv文件，转成csv文件这句程序需要运行1个小时，暂时注释掉                       
        antidp_feature, antidp_label = 0, 0 # antidp_feature, antidp_label这两个值不能空着返回

    
    for i in range(ZZHPD_drugA.shape[0]): # ZZHPD_drugA.shape[0]= 171 ZZHPD有171种组合 ###
        ZZHPD_feature.append(np.hstack((d_feature[ZZHPD_drugA[i]], d_feature[ZZHPD_drugB[i]])))    

    new_feature = np.array(new_feature)  # 323539*....
    new_label = np.array(new_label)  # 323539
    ZZHPD_feature = np.array(ZZHPD_feature)

    antidp_feature = np.array(antidp_feature)  
    antidp_label = np.array(antidp_label)  
    return new_feature, new_label, event_num, ZZHPD_feature, antidp_feature, antidp_label

def feature_vector(feature_name, df,clfName, ZZHPD_SMILESlist):
    def Jaccard(matrix):
        matrix = np.mat(matrix)

        numerator = matrix * matrix.T

        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T

        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features # 583
    '''
    # ZZHPD ###
    for i in ZZHPD_SMILESlist:
        for each_ZZHPD in i.split('|'):
            if each_ZZHPD not in all_feature:
                all_feature.append(each_ZZHPD) # 595,dataset1加了12个特征 # dataset2和ZZHPD_SMILESlist都是2040
    '''
    feature_matrix = np.zeros((len(drug_list)+len(ZZHPD_SMILESlist), len(all_feature)), dtype=float) ###
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'): # 两种方式for each_feature in i.split('|'):
            df_feature[each_feature].iloc[i] = 1
    
    # ZZHPD表填1 ###
    for i in range(len(drug_list),len(drug_list)+len(ZZHPD_SMILESlist)):
        for each_ZZHPD in ZZHPD_SMILESlist[i-len(drug_list)].split('|'): # ZZHPD_SMILESlist[0]对应的是df.iloc[572]
            if each_ZZHPD in df_feature.columns: #
                df_feature[each_ZZHPD].iloc[i] = 1      
    # df_feature总指纹为2040种，1278种药物 (1278,2040)
    if clfName == 'MDF_SA_DDI':
        df_feature = np.array(df_feature)
        sim_matrix = np.array(Jaccard(df_feature))
    
    else: # 572是DDIMDL所用数据集的药物总数，MDF所用数据集的药物总数为1258，栀子厚朴汤的药物总数为20 = 1278
        vector_size = len(np.array(df['name']).tolist())+len(ZZHPD_SMILESlist) ###
        sim_matrix = Jaccard(np.array(df_feature)) # df[smiles]_shape:(572,583)通过雅卡尔系数变成(572, 572)
        pca = PCA(n_components=vector_size)  # PCA dimension # 都压到572维 # 都压到1278维
        pca.fit(sim_matrix)
        sim_matrix = pca.transform(sim_matrix)

    return sim_matrix

def DNN(vector_size,droprate,event_num):
    train_input = Input(shape=(vector_size,), name='Inputlayer')
    train_in = Dense(512, activation='relu',kernel_initializer=glorot_normal(seed=seed))(train_input)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate,seed=seed)(train_in)
    train_in = Dense(256, activation='relu',kernel_initializer=glorot_normal(seed=seed))(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate,seed=seed)(train_in)
    train_in = Dense(event_num,kernel_initializer=glorot_normal(seed=seed))(train_in)
    out = Activation('softmax')(train_in)
    model = Model(train_input, out)
    # model = Model(input=train_input, output=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

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
    logging("train len %s" % len(y_train))
    logging("test len %s" % len(y_test))

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
        logging('epoch [%d] loss: %.6f testing_loss: %.6f ' % (
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
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    # 二分类处理稀疏矩阵用
    y_one_hot = (np.arange(y_test.max() + 1) == y_test[:, None]).astype(dtype='float32') # 真实标签形成稀疏矩阵
    pred_one_hot = (np.arange(pred_type.max() + 1) == pred_type[:, None]).astype(dtype='float32') # 真实标签形成稀疏矩阵

    # 多分类处理稀疏矩阵用label_binarize
    # y_one_hot = label_binarize(y_test, np.arange(event_num))
    # pred_one_hot = label_binarize(pred_type, np.arange(event_num))
    # 二分类
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score)
    result_all[2] = roc_auc_score(y_one_hot, pred_score)
    result_all[3] = f1_score(y_test, pred_type, average='binary')
    result_all[4] = precision_score(y_test, pred_type, average='binary')
    result_all[5] = recall_score(y_test, pred_type, average='binary')
    result_all[6], result_all[7], result_all[8], result_all[9] = confusion_matrix(y_test, pred_type).ravel() # TN, FP, FN, TP
    # 多分类
    # result_all[0] = accuracy_score(y_test, pred_type)
    # result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    # result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    # result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    # result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    # result_all[5] = f1_score(y_test, pred_type, average='micro')
    # result_all[6] = f1_score(y_test, pred_type, average='macro')
    # result_all[7] = precision_score(y_test, pred_type, average='micro')
    # result_all[8] = precision_score(y_test, pred_type, average='macro')
    # result_all[9] = recall_score(y_test, pred_type, average='micro')
    # result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel()) # 取其中一列出来，.ravel()降维
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

def cross_val(feature,label,event_num,clfName):
    skf = StratifiedKFold(n_splits=cross_ver_tim) # shuffle：默认为False；# random_state：默认为None，表示随机数的种子，只有当shuffle设置为True的时候才会生效。
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    # y_true = np.array([])
    # y_score = np.zeros((0, event_num), dtype=float)
    # y_pred = np.array([])
    matrix = []
    if type(feature) != list:
        matrix.append(feature)
        feature_matrix = matrix

    for train_index, test_index in skf.split(feature, label):
        pred = np.zeros((len(test_index), event_num), dtype=float) # shape:(7474,65)预测的标签放入此矩阵中

        for i in range(len(feature_matrix)):
            x_train, x_test = feature_matrix[i][train_index], feature_matrix[i][test_index]
            
            y_train, y_test = label[train_index], label[test_index]
            # one-hot encoding独热编码
            y_train_one_hot = np.array(y_train)
            y_train_one_hot = (np.arange(y_train_one_hot.max() + 1) == y_train[:, None]).astype(dtype='float32')
            # y_train_one_hot.max()=64，加上初始的0，即65
            # y_train[:, None] (29790,1),因为y_train = label_matrix[train_index] (29790,)，[:, None]把横躺着的(29790,)变成竖立的(29790,1)
            # np.arange(y_train_one_hot.max() + 1) == y_train[:, None]) 变成只有0和1的稀疏矩阵（29790，65）；.astype(dtype='float32') 变成0，1

            y_test_one_hot = np.array(y_test)
            y_test_one_hot = (np.arange(y_test_one_hot.max() + 1) == y_test[:, None]).astype(dtype='float32')

            logging("train len %s" % len(y_train))
            logging("test len %s" % len(y_test))

            if clfName == 'DDIMDL':
                dnn = DNN(len(feature[0]),droprate,event_num)
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
                dnn.fit(x_train, y_train_one_hot, batch_size=128, epochs=100, validation_data=(x_test, y_test_one_hot),
                        callbacks=[early_stopping])
                pred += dnn.predict(x_test) # 加在每一个数值上，之后再pred_score = pred / len(feature_matrix)进行取平均数，获得更准确的结果
                continue # 跳出本次的循环，不进入下面的clf.fit(X_train, y_train),而进入for i in range(len(feature_matrix)):

            elif clfName == 'MDF_SA_DDI':
                model = BERT(len(feature[0]),bert_n_heads,bert_n_layers,event_num) # len(feature[0])=3432
                pred = BERT_train(model,x_train,y_train,x_test,y_test,event_num)
                continue

            elif clfName == 'RF':
                clf = RandomForestClassifier(n_estimators=100)
            elif clfName == 'GBDT':
                clf = GradientBoostingClassifier()
            elif clfName == 'SVM':
                clf = SVC(probability=True)
            elif clfName == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=4)
            else:
                clf = LogisticRegression()
            clf.fit(x_train, y_train)
            pred += clf.predict_proba(x_test)

        pred_score = pred / len(feature_matrix)
        pred_type = np.argmax(pred_score, axis=1)
        # y_pred = np.hstack((y_pred, pred_type))
        # y_score = np.row_stack((y_score, pred_score)) # np.vstack和np.row_stack一样
        # y_true = np.hstack((y_true, y_test))
        # result_all, result_eve= evaluate(y_pred, y_score, y_true, event_num)
        a,b = evaluate(pred_type,pred_score,y_test,event_num)
        for i in range(all_eval_type):
            result_all[i]+=a[i]
        for i in range(each_eval_type):
            result_eve[:,i]+=b[:,i]
    result_all=result_all/5
    result_eve=result_eve/5
    
    return result_all, result_eve

# k折交叉验证，获得索引，保证所有标签均匀分布在每一折数据集中
def get_index(label_matrix, event_num, seed, CV):
    index_all_class = np.zeros(len(label_matrix)) # 标签0分成5个部分，标签1分成5个部分
    for j in range(event_num):
        index = np.where(label_matrix == j) # 判断条件，返回标签为0的所有索引（有9810），下一次返回标签为1的 
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
        k_num = 0 
        for train_index, test_index in kf.split(range(len(index[0]))): # 标签为0的索引9810分成 7848 和 1962 
            index_all_class[index[0][test_index]] = k_num # k_num=0 表示放在第0折 # 标签1的索引是在后半段，所以需要从index[0]里进行索引
            k_num += 1

    return index_all_class

# 包含k折、训练和验证的函数调用
def cross_validation(feature_matrix, label_matrix, clf_type, event_num, seed, CV): # 有37264种组合
    result_all = np.zeros((all_eval_type, 1), dtype=float) # shape：(11,1); 用np.zero可放入all_eval_type变量而且也可以遍历；当然也可用列表[0]*all_eval_type=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result_eve = np.zeros((event_num, each_eval_type), dtype=float) # shape:(65,6)
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float) # shape:(0,65)
    index_all_class = get_index(label_matrix, event_num, seed, CV)
    matrix = []
    acc=[]
    if type(feature_matrix) != list:
        matrix.append(feature_matrix)
        # =============================================================================
        #     elif len(np.shape(feature_matrix))==3:
        #         for i in range((np.shape(feature_matrix)[-1])):
        #             matrix.append(feature_matrix[:,:,i])
        # =============================================================================
        feature_matrix = matrix
    for k in range(CV):
        train_index = np.where(index_all_class != k) # 获得索引
        test_index = np.where(index_all_class == k) # k=0 7474; k=1 7464; k=2 7451; k=3 7444; k=4 7431
        pred = np.zeros((len(test_index[0]), event_num), dtype=float) # shape:(7474,65)预测的标签放入此矩阵中
        # dnn=DNN()
        for i in range(len(feature_matrix)):
            x_train = feature_matrix[i][train_index]
            x_test = feature_matrix[i][test_index]
            
            y_train = label_matrix[train_index]
            # one-hot encoding独热编码
            y_train_one_hot = np.array(y_train)
            y_train_one_hot = (np.arange(y_train_one_hot.max() + 1) == y_train[:, None]).astype(dtype='float32')
            # y_train_one_hot.max()=64，加上初始的0，即65
            # y_train[:, None] (29790,1),因为y_train = label_matrix[train_index] (29790,)，[:, None]把横躺着的(29790,)变成竖立的(29790,1)
            # np.arange(y_train_one_hot.max() + 1) == y_train[:, None]) 变成只有0和1的稀疏矩阵（29790，65）；.astype(dtype='float32') 变成0，1

            y_test = label_matrix[test_index]
            y_test_one_hot = np.array(y_test)
            y_test_one_hot = (np.arange(y_test_one_hot.max() + 1) == y_test[:, None]).astype(dtype='float32')

            if clf_type == 'DDIMDL':
                dnn = DNN(len(feature_matrix[i][train_index][0]),droprate,event_num)
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
                dnn.fit(x_train, y_train_one_hot, batch_size=128, epochs=100, validation_data=(x_test, y_test_one_hot),
                        callbacks=[early_stopping])
                pred += dnn.predict(x_test) # 加在每一个数值上，之后再pred_score = pred / len(feature_matrix)进行取平均数，获得更准确的结果
                continue # 跳出本次的循环，不进入下面的clf.fit(X_train, y_train),而进入for i in range(len(feature_matrix)):
            
            elif clf_type == 'RF':
                clf = RandomForestClassifier(n_estimators=100)
            elif clf_type == 'GBDT':
                clf = GradientBoostingClassifier()
            elif clf_type == 'SVM':
                clf = SVC(probability=True)
            elif clf_type == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=4)
            else:
                clf = LogisticRegression()
            clf.fit(x_train, y_train)
            pred += clf.predict_proba(x_test)

        pred_score = pred / len(feature_matrix)
        pred_type = np.argmax(pred_score, axis=1)

        a,b = evaluate(pred_type,pred_score,y_test,event_num)
        for i in range(all_eval_type):
            result_all[i]+=a[i]
        for i in range(each_eval_type):
            result_eve[:,i]+=b[:,i]
    result_all=result_all/5 # 5次的结果取平均，包括混淆矩阵 
    result_eve=result_eve/5

        # y_true = np.hstack((y_true, y_test)) # 总结：y_test 的5折结果保存到y_true矩阵中， np.hstack（拼接行）1.y_true空 float64 + y_test（7474，）int64 -> y_true(7474,) float64 # 2. y_true (7474,) float64 + y_test（7464，）->(14938,) 
        # y_pred = np.hstack((y_pred, pred_type)) # 总结：pred_type 的5折结果保存到y_pred矩阵中 1.y_pred空  float64+ pred_type(7474,) int64 ->y_pred(7474,) float64 # 2.y_pred (7474,) float64 + pred_type（7464，）->(14938,)
        # y_score = np.row_stack((y_score, pred_score)) # np.row_stack行合并（拼接行）,0.空+(7474,65) 1.(14938,65) 2.(22389,65) 3.(29833,65) 4.(37264,65)
    # result_all, result_eve = evaluate(y_pred, y_score, y_true, event_num, set_name) # 应该是每一折就要指标评估，之后五折求平均；而不是五折的测试集汇集起来直接进行指标评估

    return result_all, result_eve

def save_result(datasetName, featureName, result_type, clfName, result,taskname):
    with open('_cross_val_' + taskname +  '_' + clfName + '_' + result_type  + '.csv', "w", newline='',encoding='utf-8') as csvfile: # 
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

def test(train_feature, train_label, test_feature, test_label, event_num, clf_type, taskname, task_5_subtask): # task_5_subtask以1类为1，其他7类都为0
    test_all = np.zeros((all_eval_type, 1), dtype=float)
    test_eve = np.zeros((event_num, each_eval_type), dtype=float)

    pred = np.zeros((len(test_feature), event_num), dtype=float) # shape:(171,65)预测的标签放入此矩阵中

    matrix = []
    if type(train_feature) != list:
        matrix.append(train_feature)
        feature_matrix = matrix

    for i in range(len(feature_matrix)):

        feature_matrix = feature_matrix[i]

        train_label_one_hot = np.array(train_label)
        train_label_one_hot = (np.arange(train_label_one_hot.max() + 1) == train_label[:, None]).astype(dtype='float32')

        test_label_one_hot = np.array(test_label)
        test_label_one_hot = (np.arange(test_label_one_hot.max() + 1) == test_label[:, None]).astype(dtype='float32')        


        if clf_type == 'DDIMDL':
            dnn = DNN(len(feature_matrix[0]),droprate,event_num)
            early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')  #which is not available. Available metrics are: loss,accuracy
            dnn.fit(train_feature, train_label_one_hot,  batch_size=128, epochs=100,
                    callbacks=[early_stopping]) # batch_size为128和64，发现到509工作站所占的内存是差不多的，直接拉128
            pred += dnn.predict(test_feature)
            if taskname == "task5_adverse_reactions":
                dnn.save('_test_'+ str(task_5_subtask+1) +'_model_' + taskname + '_' + clf_type)
            else:
                dnn.save('_test'+'_model_' + taskname + '_' + clf_type)            
            continue
        elif clf_type == 'MDF_SA_DDI':
            model = BERT(len(feature_matrix[0]),bert_n_heads,bert_n_layers,event_num) # len(feature[0])=3432
            pred = BERT_test(model, train_feature, train_label, test_feature, event_num)
            continue
        elif clf_type == 'RF':
            clf = RandomForestClassifier(n_estimators=100)
        elif clf_type == 'GBDT':
            clf = GradientBoostingClassifier()
        elif clf_type == 'SVM':
            clf = SVC(probability=True)
        elif clf_type == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=4)
        else:
            clf = LogisticRegression()
        clf.fit(train_feature, train_label)
        pred += clf.predict_proba(test_feature)

    pred_score = pred 
    # pred_score = pred / len(feature_matrix)
    pred_type = np.argmax(pred_score, axis=1)
        
    test_all, test_eve = evaluate(pred_type,pred_score,test_label,event_num) 
    return test_all, test_eve

def save_test_result(datasetName, featureName, result_type, clfName, result,taskname):
    with open('_test_' + taskname +  '_' + clfName + '_' + result_type  + '.csv', "w", newline='',encoding='utf-8') as csvfile: # 
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

def save_test_result_task5(subclass, datasetName, featureName, result_type, clfName, result,taskname):
    with open('_test_' + subclass +  '_' +  taskname +  '_' + clfName + '_' + result_type  + '.csv', "w", newline='',encoding='utf-8') as csvfile: # 
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

def prepare_ZZHPD(drugfile, smilesfile, drugABfile,datasetName):
    # ZZHPD_druglist
    ZZHPD_druglist = []
    with open(drugfile,'r') as f:
        for line in f:
            ZZHPD_druglist.append(line.splitlines()[0]) # line:'Naringenin\n'; line.splitlines():['Naringenin']; line.splitlines()[0]:'Naringenin'

    # ZZHPD_smiles
    df_ZZHPD = pd.read_csv(smilesfile)
    smiles = df_ZZHPD['canonical_smiles'].tolist()
    # ZZHPD的smiles进行编码
    if datasetName =='dataset1': # dataset1是pubchem编码
        featurizer = dc.feat.PubChemFingerprint()
        pubchem = np.zeros((0, 881), dtype=float)
        for i in smiles:
            pubchemcoding = np.squeeze(featurizer.featurize(i)) # np.squeeze() : 去掉numpy数组冗余的维度
            pubchem = np.row_stack((pubchem,pubchemcoding)) # 行拼接
        df_ZZHPD_coding = pd.DataFrame(pubchem)
    
    else: # dataset2是morgan编码 2048
        PandasTools.AddMoleculeColumnToFrame(df_ZZHPD, smilesCol = "canonical_smiles")
        radius = 2
        nBits = 2048 # 2048
        ECFP4coding = [AllChem.GetMorganFingerprintAsBitVect(x, radius = radius, nBits = nBits) for x in df_ZZHPD['ROMol']]
        df_ZZHPD_coding = pd.DataFrame([list(l) for l in ECFP4coding]
                        # , index = MAOB.SMILES
                        , columns = [f'Morgan {i}' for i in range(nBits)]
                        )

    ZZHPD_smileslist = []
    for i in range(df_ZZHPD_coding.shape[0]): # 先行后列
        s = ''
        for j in range(df_ZZHPD_coding.shape[1]):
            if df_ZZHPD_coding.iat[i,j] == 1: # df_ZZHPD_coding.iat[i,j]，第i行第j列
                s = s + str(j) + '|'
        s = s[:-1] # 不要最后一个|
        ZZHPD_smileslist.append(s)

    
    ZZHPD_extraction = pd.read_csv(drugABfile)
    ZZHPD_drugA = ZZHPD_extraction['drugA']
    ZZHPD_drugB = ZZHPD_extraction['drugB']
    # 输出ZZHPD的药物信息
    '''
    df_ZZHPD_druginfo = pd.DataFrame()
    df_ZZHPD_druginfo['name'] = ZZHPD_druglist
    df_ZZHPD_druginfo['SMILES'] = smiles
    df_ZZHPD_druginfo['Morgan_fingerprints'] = ZZHPD_smileslist
    df_ZZHPD_druginfo.to_csv('df_ZZHPD_druginfo' + '.csv', encoding='utf-8')
    '''
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

    logging("arg train len %s" % len(y_train))

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
        logging('epoch [%d] loss: %.6f' % (epoch + 1, running_loss / len_train))

    pre_score = np.zeros((0, event_num), dtype=float)
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            inputs = data
            inputs = inputs.to(device)
            X, _, _, _ = model(inputs.float())
            pre_score = np.vstack((pre_score, F.softmax(X).cpu().numpy()))
    return pre_score

# 包含k折、训练和验证的函数调用
def predict_ZZHPD(feature_matrix, label_matrix, ZZHPD_feature, clf_type, event_num, taskname, task_5_subtask):
    pred = np.zeros((len(ZZHPD_feature), event_num), dtype=float) # shape:(171,65)预测的标签放入此矩阵中

    matrix = []
    if type(feature_matrix) != list:
        matrix.append(feature_matrix)
        # =============================================================================
        #     elif len(np.shape(feature_matrix))==3:
        #         for i in range((np.shape(feature_matrix)[-1])):
        #             matrix.append(feature_matrix[:,:,i])
        # =============================================================================
        feature_matrix = matrix

    for i in range(len(feature_matrix)):

        feature_matrix = feature_matrix[i]

        label_matrix_one_hot = np.array(label_matrix)
        label_matrix_one_hot = (np.arange(label_matrix_one_hot.max() + 1) ==label_matrix[:, None]).astype(dtype='float32')


        if clf_type == 'DDIMDL':
            dnn = DNN(len(feature_matrix[0]),droprate,event_num)
            early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')  #which is not available. Available metrics are: loss,accuracy
            dnn.fit(feature_matrix, label_matrix_one_hot, batch_size=128, epochs=100,
                    callbacks=[early_stopping],shuffle=False)
            pred += dnn.predict(ZZHPD_feature)
            # 保存模型

            if taskname == "task5_adverse_reactions":
                dnn.save('_predict_'+ str(task_5_subtask+1) +'_hzh_model_' + taskname + '_' + clf_type)
            else:
                dnn.save('_predict'+'_hzh_model_' + taskname + '_' + clf_type)
                # new_model = tf.keras.models.load_model('_model_task3_DDIMDL')
                # predict_new_model = new_model.predict(ZZHPD_feature)

            continue
        elif clf_type == 'MDF_SA_DDI':
            model = BERT(len(feature_matrix[0]),bert_n_heads,bert_n_layers,event_num) # len(feature[0])=3432
            pred = BERT_test(model, feature_matrix,label_matrix, ZZHPD_feature,event_num)
            continue
        elif clf_type == 'RF':
            clf = RandomForestClassifier(n_estimators=100)
        elif clf_type == 'GBDT':
            clf = GradientBoostingClassifier()
        elif clf_type == 'SVM':
            clf = SVC(probability=True)
        elif clf_type == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=4)
        else:
            clf = LogisticRegression()
        clf.fit(feature_matrix, label_matrix)
        pred += clf.predict_proba(ZZHPD_feature)

    pred_score = pred
    # pred_score = pred / len(feature_matrix)
    pred_type = np.argmax(pred_score, axis=1)

    return pred_type, pred_score

def classificationtask(feature_matrix, label_matrix, taskname, clfName, datasetName, featureName, ZZHPD_feature, antidp_feature): 
   
    adverse_cardiovascular = np.array([14,69,20,25,64,31,82,5,89,40,93,8,34,13,24,27,39,68,71,9,80,45,47]) # 心血管系统相关——41810
    adverse_nervous = np.array([29,63,74,61,90,21,42,77,51]) # 神经系统相关——3619
    adverse_WaterANDelectrolyte = np.array([15,70,35,87,97,75,52,86,94,99]) # 水与电解质平衡相关——4316
    adverse_blood = np.array([18,56,26,32,55,67,76,81,83]) # 血液系统相关——4827
    adverse_glycometabolism = np.array([17,19]) # 糖代谢相关——4045
    adverse_muscle = np.array([22,43,78]) # 肌肉系统相关——1907
    adverse_kidney = np.array([73,62,57,33,79,85,92,59]) # 肾脏系统相关——1639
    adverse_digestion = np.array([23]) # 消化系统——1302
    # adverse_immune = np.array([36]) # 免疫系统相关——600
    adverse_reactions = [adverse_cardiovascular,
                        adverse_nervous,
                        adverse_WaterANDelectrolyte,
                        adverse_blood,
                        adverse_glycometabolism,
                        adverse_muscle,
                        adverse_kidney,
                        adverse_digestion]

    if taskname == 'task5_adverse_reactions':
        # Negative_class = np.hstack((adverse_nervous, adverse_WaterANDelectrolyte, adverse_blood, adverse_glycometabolism, adverse_muscle, adverse_digestion, adverse_kidney)) # task5_Adverse reaction_unrelated to cardiovascular_12739_0,不必加36，test集效果差
        # Positive_class = adverse_cardiovascular # task5_Adverse reaction_related to cardiovascular_41810_1
        adverse_reactions_posi = []
        adverse_reactions_nega = []

        for i in range(len(adverse_reactions)): # 例如以心脑血管为1，其他的所有类为0
            Positive = adverse_reactions[i]
            adverse_reactions_posi.append(Positive)
            Negative = np.array([]).astype('int') # 注意一定要设置初始值并且指定类型为'int'(因为默认值为float)
            for j in range(len(adverse_reactions)):
                if j == i: # 排除Positive这一类
                    continue
                Negative = np.hstack((Negative, adverse_reactions[j]))
            adverse_reactions_nega.append(Negative)

        for subtask in range(len(adverse_reactions_posi)):
            Negative_class = adverse_reactions_nega[subtask]
            Positive_class = adverse_reactions_posi[subtask]

            Negative_class_index = np.array([]).astype('int') # 初始值一定要置0
            Positive_class_index = np.array([]).astype('int')

            for i in Negative_class:
                Negative_class_index = np.hstack((Negative_class_index,np.where(label_matrix==i)[0]))
            for j in Positive_class:
                Positive_class_index = np.hstack((Positive_class_index,np.where(label_matrix==j)[0]))
            
            task_feature_ = np.row_stack((feature_matrix[Negative_class_index], feature_matrix[Positive_class_index]))
            task_label_ = np.hstack((np.zeros(len(Negative_class_index)),np.ones(len(Positive_class_index)))).astype('int')

            train_feature_, test_feature, train_label_, test_label = train_test_split(task_feature_, task_label_, test_size = 0.2, random_state = seed) # 8/2分数据集

            logging('Original train dataset shape %s' % Counter(train_label_))
            logging('Original task dataset shape %s' % Counter(task_label_))
            sm = SMOTE(random_state=seed)
            # sm = RandomUnderSampler(random_state=seed)
            train_feature, train_label = sm.fit_resample(train_feature_, train_label_) # 训练模型
            task_feature, task_label = sm.fit_resample(task_feature_, task_label_)  # 训练预测ZZHPD的模型
            logging('Resampled train dataset shape %s' % Counter(train_label))
            logging('Resampled task dataset shape %s' % Counter(task_label))

            np.random.seed(seed) # 按同样的顺序打乱task_feature和task_label, 要两句话配套进行才能保持它的随机性 
            np.random.shuffle(task_feature)
            np.random.seed(seed)
            np.random.shuffle(task_label)

            task_event_num = len(np.unique(task_label))

            # 80/20 测试 
            df_test = pd.DataFrame()                    
            df_test.index = ["ACC", "AUPR", "AUC", "F1", "Precision", "Recall", "TN", "FP", "FN", "TP"] # df_test的索引
            for t in range(repeat_running): # 重复运行repeat_running次
                test_all, test_eve = test(train_feature, train_label, test_feature, test_label, task_event_num, clfName, taskname, subtask)
                df_test['Test_'+ str(t + 1)] = test_all[:,0] # 列名为Test_1; test_all[:,0]保持它的shape为(10, ),即值为0.99;直接test_all的shape为(10,1) ,值为array([0.99])
                df_test = pd.concat([df_test, pd.DataFrame(test_eve.T,index=["ACC", "AUPR", "AUC", "F1", "Precision", "Recall"])],axis=1) # 合并predict_ZZHPDproba
                # save_test_result(datasetName,featureName,"all",clfName,test_all,taskname)##
                # save_test_result(datasetName,featureName,"each",clfName,test_eve,taskname)##
            df_test.to_csv('_hzh_test_' + str(subtask+1) + '_' +  taskname + '_' + clfName +  '.csv', encoding='utf-8')

            # 预测predict
            # predict_ZZHPDlabel,predict_ZZHPDproba = predict_ZZHPD(task_feature, task_label, ZZHPD_feature, clfName, task_event_num)
            # df_predict_ZZHPDlabel = pd.DataFrame(predict_ZZHPDlabel).to_csv('_predict_' + str(m) + '_' +  taskname + '_' + clfName +  '.csv', encoding='utf-8')
            # df_predict_ZZHPDproba = pd.DataFrame(predict_ZZHPDproba).to_csv('_proba_' + str(m) + '_' +  taskname + '_' + clfName +  '.csv', encoding='utf-8')
            
            # antidp_val
            '''
            df_antidp_val = pd.DataFrame()
            for b in range(repeat_running): # 重复运行repeat_running次                        
                predict_antidp_val_label, predict_antidp_val_Dproba = predict_ZZHPD(task_feature, task_label, antidp_feature, clfName, task_event_num, taskname,subtask)
                df_antidp_val['Label_'+ str(b + 1)] = predict_antidp_val_label # 列名为Label_1
                df_antidp_val = pd.concat([df_antidp_val, pd.DataFrame(predict_antidp_val_Dproba)],axis=1) # 合并predict_ZZHPDproba
            df_antidp_val.to_csv('_predict_antidp_' + str(subtask+1) + '_' +  taskname + '_' + clfName +  '.csv', encoding='utf-8') # 记得有str(subtask+1)
            '''
            
            df_ZZHPD = pd.DataFrame()
            for n in range(repeat_running): # 重复运行repeat_running次                          
                predict_ZZHPDlabel, predict_ZZHPDproba = predict_ZZHPD(task_feature, task_label, ZZHPD_feature, clfName, task_event_num, taskname, subtask)
                # df_predict_ZZHPDlabel = pd.DataFrame(predict_ZZHPDlabel).to_csv('_predict_' + taskname + '_' + clfName +  '.csv', encoding='utf-8')
                # df_predict_ZZHPDproba = pd.DataFrame(predict_ZZHPDproba).to_csv('_proba_' + taskname + '_' + clfName +  '.csv', encoding='utf-8')
                df_ZZHPD['Label_'+ str(n + 1)] = predict_ZZHPDlabel # 列名为Label_1
                # df_ZZHPDproba = pd.DataFrame(predict_ZZHPDproba,columns=['ProbaZero_' + str(n + 1), 'ProbaOne_' + str(n + 1)])
                df_ZZHPD = pd.concat([df_ZZHPD, pd.DataFrame(predict_ZZHPDproba)],axis=1) # 合并predict_ZZHPDproba
            df_ZZHPD.to_csv('_hzh_predict_' + str(subtask+1) + '_' +  taskname + '_' + clfName +  '.csv', encoding='utf-8')


    else:       
    # task1---metabolism; task2---serum_concentration; task3---therapeutic_efficacy; task4---nervous system effectiveness;
    # task5_cardiovascular_hypertensionORqTc---Cardiovascular_Adverse_Reactions; task5_nervous_WaterANDelectrolyte--- Nervous system adverse reactions
        if taskname == 'task1': # metabolism ###
            Negative_class = np.array([3]) # task1_metabolism_increase_26509_0
            Positive_class = np.array([0]) # task1_metabolism_decrease_100725_1
        elif taskname == 'task2': # serum_concentration ###
            Negative_class = np.array([11])  # task2_serum_concentration_decrease_4207_0
            Positive_class = np.array([4]) # task2_serum_concentration_increase_16588_1
        elif taskname == 'task3': # therapeutic_efficacy ###
            Negative_class = np.array([12]) # task3_therapeutic_efficacy_increase_3751_0
            Positive_class = np.array([6])  # task3_therapeutic_efficacy_decrease_12447_1
        elif taskname == 'task4': # nervous system effectiveness ###
            Negative_class = np.array([28, 53, 44, 50, 66, 48, 54, 88, 60, 65]) # task4_UnToNerve_system_3235_0
            Positive_class = np.array([7, 41, 46, 58, 72, 84, 37, 98, 38]) # task4_ToNerve_system_11650_1
        elif taskname == 'task5_cardiovascular_hypertensionORqTc': # Cardiovascular_Adverse_Reactions ###
            Negative_class = np.array([5,89,40,93]) # task5_cardiovascular_QTc_16868_0    
            Positive_class = np.array([8,34,13,24,27,39,68,71,9,80,45,47]) # task5_cardiovascular_hypertension_18077_1        
        elif taskname == 'task5':
            Negative_class = np.hstack((adverse_nervous, adverse_WaterANDelectrolyte, adverse_blood, adverse_glycometabolism, adverse_muscle, adverse_digestion, adverse_kidney)) # task5_Adverse reaction_unrelated to cardiovascular_12739_0,不必加36，test集效果差
            Positive_class = adverse_cardiovascular # task5_Adverse reaction_related to cardiovascular_41810_1
        elif taskname == 'task5_1':
            Negative_class = np.hstack((adverse_WaterANDelectrolyte, adverse_blood, adverse_glycometabolism, adverse_muscle, adverse_digestion, adverse_kidney)) # task5_1 Adverse reaction_unrelated to cardiovascular and nervous _18036_0
            Positive_class = adverse_nervous # task5_1 Adverse reaction_related to nervous_12739_1
        elif taskname == 'task5_2':
            Negative_class = np.hstack((adverse_glycometabolism, adverse_muscle, adverse_digestion, adverse_kidney)) # task5_2 Adverse reaction_related to glycometabolism, muscle, digestion, kidney_8893_0
            Positive_class = np.hstack((adverse_WaterANDelectrolyte, adverse_blood)) # task5_2 Adverse reaction_related to adverse_WaterANDelectrolyte, adverse_blood_9143_1
        elif taskname == 'task5_3':
            Negative_class = adverse_blood # task5_3 Adverse reaction_related to blood_4827_0
            Positive_class = adverse_WaterANDelectrolyte # task5_3 Adverse reaction_related to WaterANDelectrolyte_4316_1        
        elif taskname == 'task5_4':
            Negative_class = np.hstack((adverse_muscle, adverse_digestion, adverse_kidney)) # task5_4 Adverse reaction_related to muscle, digestion, kidney_4848_0
            Positive_class = adverse_glycometabolism # task5_4 Adverse reaction_related to glycometabolism_4045_1
        elif taskname == 'task5_5':
            Negative_class = np.hstack((adverse_digestion, adverse_kidney)) # task5_5 Adverse reaction_related to digestion, kidney_2941_1
            Positive_class = adverse_muscle # task5_5 Adverse reaction_related to muscle_1907_1
        elif taskname == 'task5_6':
            Negative_class = adverse_kidney # task5_6 Adverse reaction_related to kidney_1346_0
            Positive_class = adverse_digestion # task5_6 Adverse reaction_related to digestion_1595_1
        elif taskname == 'task5_cardiovascular':
            Negative_class = np.array([14,69,20,25,64,31,82,5,89,40,93]) # task5_cardiovascular_QTcANDarrhythmogenic_23733_0
            Positive_class = np.array([8,34,13,24,27,39,68,71,9,80,45,47]) # task5_cardiovascular_hypertension_18077_1
        elif taskname == 'task5_cardiovascular_qOr': # task5_cardiovascular_QTcORarrhythmogenic 心血管系统的心动、心律还是QTc 
            Negative_class = np.array([14,69,20,25,64,31,82]) # task5_cardiovascular_arrhythmogenic_3014_0
            Positive_class = np.array([5,89,40,93]) # task5_cardiovascular_QTc_16229_1
        elif taskname == 'task5_nervous_WaterANDelectrolyte':  # Nervous system adverse reactions ######
            # Negative_class = np.array([17,19]) # 糖代谢相关——4045
            Negative_class = np.array([15,70,35,87,97,75,52,86,94,99]) # 水与电解质平衡相关——4316_0
            Positive_class = np.array([29,63,74,61,90,21,42,77,51]) # 神经系统相关——3619_1
            
        elif taskname == 'task5_digestion':
            Negative_class = np.array([23]) # task5_digestion_gastrointestinal_1302_0
            Positive_class = np.array([73,62]) # # task5_digestion_liver_192_1  
        else:
            print(1)      

        Negative_class_index = np.array([]).astype('int')
        Positive_class_index = np.array([]).astype('int')

        for i in Negative_class:
            Negative_class_index = np.hstack((Negative_class_index,np.where(label_matrix==i)[0]))
        for j in Positive_class:
            Positive_class_index = np.hstack((Positive_class_index,np.where(label_matrix==j)[0]))
        
        task_feature_ = np.row_stack((feature_matrix[Negative_class_index], feature_matrix[Positive_class_index]))
        task_label_ = np.hstack((np.zeros(len(Negative_class_index)),np.ones(len(Positive_class_index)))).astype('int')

        train_feature_, test_feature, train_label_, test_label = train_test_split(task_feature_, task_label_, test_size = 0.2, random_state = seed)

        logging('Original train dataset shape %s' % Counter(train_label_))
        logging('Original task dataset shape %s' % Counter(task_label_))
        sm = SMOTE(random_state=seed)
        # sm = RandomUnderSampler(random_state=seed)
        train_feature, train_label = sm.fit_resample(train_feature_, train_label_)
        task_feature, task_label = sm.fit_resample(task_feature_, task_label_)
        logging('Resampled train dataset shape %s' % Counter(train_label))
        logging('Resampled task dataset shape %s' % Counter(task_label))

        np.random.seed(seed) # 按同样的顺序打乱new_feature和new_label, 要两句话配套进行才能保持它的随机性 
        np.random.shuffle(task_feature)
        np.random.seed(seed)
        np.random.shuffle(task_label)

        task_event_num = len(np.unique(task_label))
    return task_feature, task_label, train_feature, train_label, test_feature, test_label, task_event_num

def main(args):
    
    feature_list = args['featureList']
    featureName="+".join(feature_list) 

    for datasetName in args['dataset']: # 取字符串'dataset1'而不是['dataset1']

        ZZHPD_Druglist, ZZHPD_SMILESlist, ZZHPD_drugA, ZZHPD_drugB = prepare_ZZHPD('DDIMDL/datas/GMDZT_SM_drug.txt','DDIMDL/datas/GMDZT_SM_smlies.csv','DDIMDL/datas/GMDZT_SM_DrugAB.csv',datasetName)

        if datasetName =='dataset1': 
            conn = sqlite3.connect("./event.db")
            df_drug = pd.read_sql('select * from drug;', conn)
            extraction = pd.read_sql('select * from extraction;', conn)
            mechanism = extraction['mechanism']
            action = extraction['action']
            drugA = extraction['drugA']
            drugB = extraction['drugB']        
        else: # dataset2
            df_drug = pd.read_csv('DDIMDL/drug_information_del_noDDIxiaoyu50.csv',index_col=0)
            extraction = pd.read_csv('DDIMDL/df_extraction_cleanxiaoyu50.csv',index_col=0)
            mechanism = extraction['mechanism']
            action = extraction['action']
            drugA = extraction['drugA']
            drugB = extraction['drugB']
    
        
        for clfName in args['classifier']:
            print(1)
            new_feature, new_label, event_num, ZZHPD_feature,antidp_feature, antidp_label = prepare(df_drug,feature_list,mechanism,action,drugA,drugB,clfName,ZZHPD_Druglist, ZZHPD_SMILESlist, ZZHPD_drugA, ZZHPD_drugB, args['antidp_val'][0])
            print(2)
            # 虽然有for循环，但是new_label每次都是对drugAdrugB的所有事件进行标签编码
            # 当antidp_val为YES时，抽出antidp_feature
            np.random.seed(seed) # 按同样的顺序打乱new_feature和new_label, 要两句话配套进行才能保持它的随机性 
            np.random.shuffle(new_feature)
            np.random.seed(seed)
            np.random.shuffle(new_label)

            for taskname in args['task']:
                start=time.time()
                logging("repeat_running: %s" %(repeat_running)) # 一个任务需要重复运行几次
                logging("clfName: %s; taskname: %s" %(clfName, taskname)) # 一个任务运行的时间
                if taskname == "task5_adverse_reactions": # 单独运行80/20和预测ZZHPD
                    classificationtask(new_feature, new_label, taskname, clfName, datasetName, featureName, ZZHPD_feature, antidp_feature)
                else:
                    task_5_subtask = 0 
                    task_feature, task_label, train_feature, train_label, test_feature, test_label, task_event_num = classificationtask(new_feature, new_label, taskname, clfName, datasetName, featureName, ZZHPD_feature, antidp_feature)
                    # 下面用的是task_event_num = 2，而不是event_num = 100
                    # 五倍交叉验证
                    # result_all, result_eve = cross_val(train_feature, train_label, task_event_num, clfName)
                    # save_result(datasetName,featureName,"all",clfName,result_all,taskname)##
                    # save_result(datasetName,featureName,"each",clfName,result_eve,taskname)##
 
                    # 80/20 测试                                  
                    df_test = pd.DataFrame()                    
                    df_test.index = ["ACC", "AUPR", "AUC", "F1", "Precision", "Recall", "TN", "FP", "FN", "TP"] # df_test的索引
                    for t in range(repeat_running): # 重复运行repeat_running次
                        test_all, test_eve = test(train_feature, train_label, test_feature, test_label, task_event_num, clfName, taskname, task_5_subtask)
                        df_test['Test_'+ str(t + 1)] = test_all[:,0] # 列名为Test_1; test_all[:,0]保持它的shape为(10, ),即值为0.99;直接test_all的shape为(10,1) ,值为array([0.99])
                        df_test = pd.concat([df_test, pd.DataFrame(test_eve.T,index=["ACC", "AUPR", "AUC", "F1", "Precision", "Recall"])],axis=1) # 合并predict_ZZHPDproba
                        # save_test_result(datasetName,featureName,"all",clfName,test_all,taskname)##
                        # save_test_result(datasetName,featureName,"each",clfName,test_eve,taskname)##
                    df_test.to_csv('DDIMDL/GMDZT_SM/_GMDZT_SM_test_' + taskname + '_' + clfName +  '.csv', encoding='utf-8')
                                          
                    # antidp_val，为什么不用evaluate(pred_type, pred_score, y_test, event_num)得到准确值呢，主要是因为有些没有标签0，有些没有标签1，所以就手动统计了
                    '''   
                    df_antidp_val = pd.DataFrame()
                    for b in range(repeat_running): # 重复运行repeat_running次                        
                        predict_antidp_val_label, predict_antidp_val_Dproba = predict_ZZHPD(task_feature, task_label, antidp_feature, clfName, task_event_num, taskname,task_5_subtask)
                        df_antidp_val['Label_'+ str(b + 1)] = predict_antidp_val_label # 列名为Label_1
                        df_antidp_val = pd.concat([df_antidp_val, pd.DataFrame(predict_antidp_val_Dproba)],axis=1) # 合并predict_ZZHPDproba
                    df_antidp_val.to_csv('_predict_antidp_' + taskname + '_' + clfName +  '.csv', encoding='utf-8')                    
                    '''   

                    # 预测ZZHPD
                                       
                    df_ZZHPD = pd.DataFrame()
                    for n in range(repeat_running): # 重复运行repeat_running次                        
                        predict_ZZHPDlabel, predict_ZZHPDproba = predict_ZZHPD(task_feature, task_label, ZZHPD_feature, clfName, task_event_num, taskname,task_5_subtask)
                        # df_predict_ZZHPDlabel = pd.DataFrame(predict_ZZHPDlabel).to_csv('_predict_' + taskname + '_' + clfName +  '.csv', encoding='utf-8')
                        # df_predict_ZZHPDproba = pd.DataFrame(predict_ZZHPDproba).to_csv('_proba_' + taskname + '_' + clfName +  '.csv', encoding='utf-8')
                        df_ZZHPD['Label_'+ str(n + 1)] = predict_ZZHPDlabel # 列名为Label_1
                        # df_ZZHPDproba = pd.DataFrame(predict_ZZHPDproba,columns=['ProbaZero_' + str(n + 1), 'ProbaOne_' + str(n + 1)])
                        df_ZZHPD = pd.concat([df_ZZHPD, pd.DataFrame(predict_ZZHPDproba)],axis=1) # 合并predict_ZZHPDproba
                    df_ZZHPD.to_csv('DDIMDL/GMDZT_SM/_GMDZT_SM_predict_' + taskname + '_' + clfName +  '.csv', encoding='utf-8')
                                        
                logging("time used: %s min" %((time.time() - start) / 60))

if __name__ == "__main__":
    # 默认dataset2和smile
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d","--dataset",choices=["dataset1","dataset2"],default=["dataset2"],help="datasets to use",nargs="+")
    parser.add_argument("-f","--featureList",choices=["smile","target","enzyme"],default=["smile"],help="features to use",nargs="+")
    parser.add_argument("-c","--classifier",choices=["MDF_SA_DDI","DDIMDL","RF","GBDT","SVM","KNN","LR"],default=["DDIMDL"],help="classifiers to use",nargs="+")
    parser.add_argument("-t","--task",choices=["task1","task2","task3","task4","task5_cardiovascular_hypertensionORqTc","task5_nervous_WaterANDelectrolyte"                                
                                                ,"task5","task5_1","task5_2","task5_3","task5_4","task5_5","task5_6","task5_adverse_reactions"
                                                ,"task5_cardiovascular","task5_cardiovascular_qOr","task5_digestion"
                                                ],default=["task1","task2","task3","task4","task5_adverse_reactions"]                                      
                ,help="1:metabolism; 2:serum concentration; 3:therapeutic efficacy; 4:whether related to nerve system; 5.adverse reactions",nargs="+")
    # task1---metabolism; task2---serum_concentration; task3---therapeutic_efficacy; task4---nervous system effectiveness;
    # task5_cardiovascular_hypertensionORqTc---Cardiovascular_Adverse_Reactions; task5_nervous_WaterANDelectrolyte--- Nervous system adverse reactions
    parser.add_argument("-v","--antidp_val", choices=["Yes", "No"],default=["No"],help="antidepressant_validation",nargs="+") 
    # 是否需要验证抗抑郁药物间的实验，只进行80/20和预测ZZHPD时需要把antidp_val改成NO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    args=vars(parser.parse_args())
    logging(args)
    main(args)    

'''           
            else:
                # get_index比StratifiedKFold的切分方式可以获得好一点点的性能(波动上下0.2)，可以是随机种子不一样
                ## get_index切分数据集
                for taskname in args['task']:
                    new_feature, new_label, event_num, ZZHPD_feature = prepare(df_drug, feature_list, mechanism,action,drugA,drugB,clfName,ZZHPD_Druglist, ZZHPD_SMILESlist, ZZHPD_drugA, ZZHPD_drugB)
                    task_feature, task_label, task_event_num = classificationtask(new_feature, new_label, taskname)
                    
                    start=time.time()
                    result_all, result_eve=cross_validation(task_feature, task_label, clfName, task_event_num, seed=0, CV=5) ##
                    
                    ## StratifiedKFold切分数据集
                    # for feature in feature_list:
                    #     new_feature, new_label, event_num = prepare(df_drug, [feature], mechanism,action,drugA,drugB,clfName)

                    #     np.random.seed(seed) # 按同样的顺序打乱new_feature和new_label, 要两句话配套进行才能保持它的随机性 
                    #     np.random.shuffle(new_feature)
                    #     np.random.seed(seed)
                    #     np.random.shuffle(new_label)

                    #     all_matrix.append(new_feature)            

                    # start=time.time()
                    # result_all, result_eve=cross_val(all_matrix, new_feature,new_label,event_num,clfName)
                    # print("time used:", (time.time() - start) / 3600)
                    
                    # save_result(datasetName,featureName,"all",clfName,result_all,taskname)##
                    # save_result(datasetName,featureName,"each",clfName,result_eve,taskname)##

                    predict_ZZHPDlabel = predictDDIMDL_ZZHPD(task_feature, task_label, ZZHPD_feature, clfName, task_event_num)
                    df_predict_ZZHPDlabel = pd.DataFrame(predict_ZZHPDlabel).to_csv('predict' + '_' + taskname + '_0' + clfName +  '.csv', encoding='utf-8')
                    print("time used:", (time.time() - start) / 3600)
''' 





