import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from NLPProcess import NLPProcess
# from numpy.random import seed # 
# seed(1)
import deepchem as dc

#from tensorflow import set_random_seed
#set_random_seed(2)
import csv
import sqlite3
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
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
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Model  #
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, BatchNormalization #
from tensorflow.keras.callbacks import EarlyStopping #

# import os


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True      #程序按需申请内存
# sess = tf.compat.v1.Session(config = config)


event_num = 65
droprate = 0.3
vector_size = 572 + 19 # 向量的长度为572+19=591 vector_size = 572 + len(ZZHPD_SMILESlist)

## DNN模型
def DNN():
    train_input = Input(shape=(vector_size * 2,), name='Inputlayer')
    train_in = Dense(512, activation='relu')(train_input)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(256, activation='relu')(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(event_num)(train_in)
    out = Activation('softmax')(train_in)
    model = Model(train_input, out)
    # model = Model(input=train_input, output=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# 准备药物与药物相互作用的特征和标签
# 这里会产生三个特征矩阵（其中，每一行为一个药物的向量），并获得两个药物拼接在一起的向量之后形成一个矩阵（其中，每一行为两个药物的向量）
def prepare(df_drug, feature_list, vector_size,mechanism,action,drugA,drugB,ZZHPD_Druglist, ZZHPD_SMILESlist, ZZHPD_drugA, ZZHPD_drugB): # 
    d_label = {}
    d_feature = {}
    # Transfrom the interaction event to number
    # Splice the features
    d_event=[]
    for i in range(len(mechanism)): # len(mechanism) = 37264
        d_event.append(mechanism[i]+" "+action[i]) # ['The risk or severity of adverse effects increase']
    label_value = 0
    count={}
    for i in d_event: # 计算d_event的频率
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    list1 = sorted(count.items(), key=lambda x: x[1],reverse=True) 
    # 对d_event的频率进行排序。count.items() 为待排序的对象；key=lambda x: x[1] 为对前面的对象中的第二维数据（即value）的值进行排序，即对事件的多少来排序，reverse=True从大到小进行排序
    
    for i in range(len(list1)): # [['The metabolism decrease', 9810],['The serum concentration increase', 5646]]
        d_label[list1[i][0]]=i # 对The metabolism decrease进行打标签为0，The serum concentration increase打标签为1

    vector = np.zeros((len(np.array(df_drug['name']).tolist())+len(ZZHPD_SMILESlist), 0), dtype=float) # 0矩阵，array([], shape=(572, 0), dtype=float64)
    for i in feature_list: # feature_list =['smile']/['target']/['enzyme]
        vector = np.hstack((vector, feature_vector(i, df_drug, vector_size, ZZHPD_SMILESlist))) #  np.hstack（拼接列）# np.hstack将参数元组的元素数组按水平方向进行叠加 （572，0），（572，572）->(572,572) ;例如：（572，1），（572，572）->(572,573)


    for i in range(len(np.array(df_drug['name']).tolist())): # 获取每一种药物的矩阵
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i] # 'Glucosamine' = 特征矩阵的第0行，添加如d_feature字典中


    for i in range(len(np.array(df_drug['name']).tolist()),len(np.array(df_drug['name']).tolist())+len(ZZHPD_SMILESlist)):
        d_feature[ZZHPD_Druglist[i-len(np.array(df_drug['name']).tolist())]] = vector[i] # 'Naringenin' = vector[572]

    new_feature = []
    new_label = []
    ZZHPD_feature = []
    name_to_id = {}
    for i in range(len(d_event)):
        new_feature.append(np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))) # 拼接两种药物的矩阵
        new_label.append(d_label[d_event[i]])

    for i in range(ZZHPD_drugA.shape[0]): # ZZHPD_drugA.shape[0]= 171 ZZHPD有171种组合
        ZZHPD_feature.append(np.hstack((d_feature[ZZHPD_drugA[i]], d_feature[ZZHPD_drugB[i]])))

    ZZHPD_feature = np.array(ZZHPD_feature)
    new_feature = np.array(new_feature)
    new_label = np.array(new_label)
    return (new_feature, new_label, event_num, ZZHPD_feature)

## 三个特征矩阵并经过Jaccard和PCA到572维
def feature_vector(feature_name, df, vector_size, ZZHPD_SMILESlist):
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

    feature_matrix = np.zeros((len(drug_list)+len(ZZHPD_SMILESlist), len(all_feature)), dtype=float) # df[smiles]_shape:(572+19,583)，df[target]_shape:(572,1162)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # 构成特征矩阵，列名为各不同的特征

    # 为特征矩阵空表填1
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'): # 为特征矩阵进行赋值1
            df_feature[each_feature].iloc[i] = 1  #  df_feature[9].iloc[0] # 特征为9的列+第0行进行赋值
    
    # ZZHPD表填1
    for i in range(len(drug_list),len(drug_list)+len(ZZHPD_SMILESlist)):
        for each_ZZHPD in ZZHPD_SMILESlist[i-len(drug_list)].split('|'): # ZZHPD_SMILESlist[0]对应的是df.iloc[572]
            df_feature[each_ZZHPD].iloc[i] = 1  

    sim_matrix = Jaccard(np.array(df_feature)) # df[smiles]_shape:(572,583)通过雅卡尔系数变成(572+19, 572+19)= (591,591)
    sim_matrix1 = np.array(sim_matrix) # shape:(591,591)
    count = 0
    pca = PCA(n_components=vector_size)  # PCA dimension # 都压到572维
    pca.fit(sim_matrix)
    sim_matrix = pca.transform(sim_matrix)
    return sim_matrix

# k折交叉验证，获得索引，保证所有标签均匀分布在每一折数据集中
def get_index(label_matrix, event_num, seed, CV):
    index_all_class = np.zeros(len(label_matrix))
    for j in range(event_num):
        index = np.where(label_matrix == j) # 判断条件，返回标签为0的所有索引（有9810），下一次返回标签为1的 
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
        k_num = 0 
        for train_index, test_index in kf.split(range(len(index[0]))): # 标签为0的索引9810分成 7848 和 1962 
            index_all_class[index[0][test_index]] = k_num # k_num=0 表示放在第0折
            k_num += 1

    return index_all_class

# 包含k折、训练和验证的函数调用
def cross_validation(feature_matrix, label_matrix, clf_type, event_num, seed, CV, set_name): # 有37264种组合
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float) # shape：(11,1); 用np.zero可放入all_eval_type变量而且也可以遍历；当然也可用列表[0]*all_eval_type=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float) # shape:(65,6)
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float) # 先定义多少列，有65类，无行--->>>shape:(0,65)
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
        test_index = np.where(index_all_class == k) # k=0 7474; k=1 7474; k=2 7451; k=3 7444; k=4 7431
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
                dnn = DNN()
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
            elif clf_type == 'FM':
                clf = GradientBoostingClassifier()
            elif clf_type == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=4)
            else:
                clf = LogisticRegression()
            clf.fit(x_train, y_train)
            pred += clf.predict_proba(x_test)

        pred_score = pred / len(feature_matrix)
        pred_type = np.argmax(pred_score, axis=1)

        a,b = evaluate(pred_type,pred_score,y_test,event_num,set_name)
        for i in range(all_eval_type):
            result_all[i]+=a[i]
        for i in range(each_eval_type):
            result_eve[:,i]+=b[:,i]
    result_all=result_all/5
    result_eve=result_eve/5

        # y_true = np.hstack((y_true, y_test)) # 总结：y_test 的5折结果保存到y_true矩阵中， np.hstack（拼接行）1.y_true空 float64 + y_test（7474，）int64 -> y_true(7474,) float64 # 2. y_true (7474,) float64 + y_test（7464，）->(14938,) 
        # y_pred = np.hstack((y_pred, pred_type)) # 总结：pred_type 的5折结果保存到y_pred矩阵中 1.y_pred空  float64+ pred_type(7474,) int64 ->y_pred(7474,) float64 # 2.y_pred (7474,) float64 + pred_type（7464，）->(14938,)
        # y_score = np.row_stack((y_score, pred_score)) # np.row_stack行合并（拼接行）,0.空+(7474,65) 1.(14938,65) 2.(22389,65) 3.(29833,65) 4.(37264,65)
    # result_all, result_eve = evaluate(y_pred, y_score, y_true, event_num, set_name) # 应该是每一折就要指标评估，之后五折求平均；而不是五折的测试集汇集起来直接进行指标评估

    return result_all, result_eve

# 把5折结果（37264，65）一起进行总的验证结果，并没有分开验证每一折的结果的11个指标和每一类的验证结果的6个指标
def evaluate(pred_type, pred_score, y_test, event_num, set_name):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float) # (11,1)，存储下面11中结果
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float) # (65,6)
    y_one_hot = label_binarize(y_test, np.arange(event_num)) # 真实标签形成稀疏矩阵 # （37264，65） # label_binarize适用于多分类，二分类应用还是一维不会变成独热编码
    pred_one_hot = label_binarize(pred_type, np.arange(event_num)) # 预测标签形成稀疏矩阵 # （37264，65）

    precision, recall, th = multiclass_precision_recall_curve(y_one_hot, pred_score)

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
    for i in range(event_num): # 有65类，用6个性能指标衡量把每一类的预测对的结果
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



def self_metric_calculate(y_true, pred_type):
    y_true = y_true.ravel()
    y_pred = pred_type.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_pred_c = y_pred.take([0], axis=1).ravel()
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(len(y_true_c)):
        if (y_true_c[i] == 1) and (y_pred_c[i] == 1):
            TP += 1
        if (y_true_c[i] == 1) and (y_pred_c[i] == 0):
            FN += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 1):
            FP += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 0):
            TN += 1
    print("TP=", TP, "FN=", FN, "FP=", FP, "TN=", TN)
    return (TP / (TP + FP), TP / (TP + FN))


def multiclass_precision_recall_curve(y_true, y_score):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_score_c = y_score.take([0], axis=1).ravel()
    precision, recall, pr_thresholds = precision_recall_curve(y_true_c, y_score_c)
    return (precision, recall, pr_thresholds)


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision, reorder=True)

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

def drawing(d_result, contrast_list, info_list):
    column = []
    for i in contrast_list:
        column.append(i)
    df = pd.DataFrame(columns=column)
    if info_list[-1] == 'aupr':
        for i in contrast_list:
            df[i] = d_result[i][:, 1]
    else:
        for i in contrast_list:
            df[i] = d_result[i][:, 2]
    df = df.astype('float')
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    df.plot.box(ylim=[0, 1.0], grid=True, color=color)
    return 0


def save_result(feature_name, result_type, clf_type, result):
    with open('ZZHPD_' + feature_name + '_' + result_type + '_' + clf_type+ '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i) # 一行一行写，先写00这一行，再写01这一行。。。
    return 0 # return 0 ： 说明程序正常退出，返回到主程序继续往下执行

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

# 包含k折、训练和验证的函数调用
def predict_ZZHPD(feature_matrix, label_matrix, clf_type, ZZHPD_feature):
    pred = np.zeros((len(ZZHPD_feature), event_num), dtype=float) # shape:(171,65)预测的标签放入此矩阵中

    for i in range(len(feature_matrix)):

        feature_matrix = feature_matrix[i]

        label_matrix_one_hot = np.array(label_matrix)
        label_matrix_one_hot = (np.arange(label_matrix_one_hot.max() + 1) ==label_matrix[:, None]).astype(dtype='float32')


        if clf_type == 'DDIMDL':
            dnn = DNN()
            early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')  #which is not available. Available metrics are: loss,accuracy
            dnn.fit(feature_matrix, label_matrix_one_hot, batch_size=128, epochs=100,
                    callbacks=[early_stopping])
            pred += dnn.predict(ZZHPD_feature)
            continue

        
        elif clf_type == 'RF':
            clf = RandomForestClassifier(n_estimators=100)
        elif clf_type == 'GBDT':
            clf = GradientBoostingClassifier()
        elif clf_type == 'SVM':
            clf = SVC(probability=True)
        elif clf_type == 'FM':
            clf = GradientBoostingClassifier()
        elif clf_type == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=4)
        else:
            clf = LogisticRegression()
        clf.fit(feature_matrix, label_matrix)
        pred += clf.predict_proba(ZZHPD_feature)

    pred_score = pred / len(feature_matrix)
    pred_type = np.argmax(pred_score, axis=1)

    return pred_type


def main(args):
    seed = 0
    CV = 5
    interaction_num = 10
    conn = sqlite3.connect("event.db")
    df_drug = pd.read_sql('select * from drug;', conn) # event中的drug表
    df_event = pd.read_sql('select * from event_number;', conn) # event中的event_number表
    df_interaction = pd.read_sql('select * from event;', conn) # event中的event表

    feature_list = args['featureList'] # smile + target + enzyme
    featureName="+".join(feature_list)
    clf_list = args['classifier'] # DDIMDL
    for feature in feature_list:
        set_name = feature + '+'
    set_name = set_name[:-1] # 这里只剩"enzyme",其实后面没有什么作用
    result_all = {}
    result_eve = {}
    all_matrix = []
    drugList=[]
    for line in open("DrugList.txt",'r'): # line.split()就是'Abemaciclib', line.split()[0]就是Abemaciclib
        drugList.append(line.split()[0])
    if args['NLPProcess']=="read":
        extraction = pd.read_sql('select * from extraction;', conn)
        mechanism = extraction['mechanism']
        action = extraction['action']
        drugA = extraction['drugA']
        drugB = extraction['drugB']
    else:
        mechanism,action,drugA,drugB=NLPProcess(drugList,df_interaction)

    ZZHPD_Druglist, ZZHPD_SMILESlist, ZZHPD_drugA, ZZHPD_drugB = ZZHPD('ZZHPD_drug.txt','ZZHPD_smlies.csv','ZZHPD_DrugAB.csv')

    

    for feature in feature_list:
        print(feature)
        new_feature, new_label, event_num, ZZHPD_feature = prepare(df_drug, [feature], vector_size, mechanism,action,drugA,drugB,ZZHPD_Druglist, ZZHPD_SMILESlist, ZZHPD_drugA, ZZHPD_drugB) # 用到event中的drug表和extraction表
        all_matrix.append(new_feature)

    start = time.clock()

    for clf in clf_list:
        print(clf)
        all_result, each_result = cross_validation(all_matrix, new_label, clf, event_num, seed, CV,
                                                   set_name)
        # =============================================================================
        #     save_result('all_nosim','all',clf,all_result)
        #     save_result('all_nosim','eve',clf,each_result)
        # =============================================================================
        save_result(featureName, 'all', clf, all_result)
        save_result(featureName, 'each', clf, each_result)
        result_all[clf] = all_result
        result_eve[clf] = each_result

        predict_ZZHPDlabel = predict_ZZHPD(all_matrix, new_label, clf, ZZHPD_feature)
    df_predict_ZZHPDlabel = pd.DataFrame(predict_ZZHPDlabel).to_csv("./predictDDIMDL_ZZHPDlabel.csv", encoding='gbk') 

    print("time used:", time.clock() - start)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f","--featureList",choices=["smile","target","enzyme"],default=["smile"],help="features to use",nargs="+")
    parser.add_argument("-c","--classifier",choices=["DDIMDL","RF","KNN","LR"],default=["DDIMDL"],help="classifiers to use",nargs="+")
    parser.add_argument("-p","--NLPProcess",choices=["read","process"],default="read",help="Read the NLP extraction result directly or process the events again")
    args=vars(parser.parse_args())
    print(args)
    main(args)


