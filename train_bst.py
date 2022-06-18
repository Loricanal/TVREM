from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score as f1_score_func
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import itertools
import pandas as pd
import ml_metrics as metrics
import pickle
import sys
import numpy as np
import os
import json
import random
from datetime import datetime
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import  VotingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler 
import xgboost as xgb

plst = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  # 多分类的问题
    'gamma': 1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth':5,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.8,              # 随机采样训练样本
    'colsample_bytree': 0.8,       # 生成树时进行的列采样
    'min_child_weight': 5,
    'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.001,                  # 如同学习率
    'seed': 1000,
    'nthread': 4,                  # cpu 线程数
}


try:
    input_folder = sys.argv[1]
except:
    raise("You have to specify the input folder")
try:
    output_folder = sys.argv[2]
except:
    raise("You have to specify the output folder")

try:
    os.makedirs(output_folder)
except:
    pass

from keras.utils import Sequence
class DataGenerator(Sequence):
    def __init__(self, train_df, train_index, batch_per_epoch=1, n=1, even=True):
        self.cols = [c for c in list(train_df.columns) if c != 'Y']
        self.train_index_1 = train_index[train_index['value'] == 1]
        self.train_index_0 = train_index[train_index['value'] == 0]
        #self.index_1 = set(self.train_index_1.index)
        #self.index_0 = set(self.train_index_0.index)
        self.train_1_df = train_df[train_df['Y']>0.1]
        self.train_0_df = train_df[train_df['Y']<0.1]
        #self.train_0_df_cp = self.train_0_df.copy()
        self.n = n
        self.batch_per_epoch = batch_per_epoch
        self.even = even
    def __len__(self):
        return int(self.batch_per_epoch)
    def __getitem__(self, index):
        return self._generate()
    def _generate(self):
        ids = list(self.train_index_1['video'])
        for j,id_ in enumerate(ids):
            train_index_0_s = self.train_index_0[self.train_index_0['video'] == id_]
            train_index_0_s = train_index_0_s.sample(n=self.n)
            inds = train_index_0_s.index
            train_index_0_s = self.train_0_df[self.train_0_df.index.isin(inds)]
            if j == 0:
                train_0_df_cp = train_index_0_s
            else:
                train_0_df_cp = pd.concat([train_0_df_cp,train_index_0_s])

        train_1_df = self.train_1_df
        if self.even:
            for i in range(1,self.n):
                train_1_df = pd.concat([train_1_df,self.train_1_df]).sample(frac=1)
        sample = pd.concat([train_1_df,train_0_df_cp])
        sample = sample.sample(frac=1)
        X = sample[self.cols].values
        Y = sample["Y"]
        return X,Y


def generateGridList(my_dict):
    keys, values = zip(*my_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return pd.DataFrame(permutations_dicts).drop_duplicates().to_dict(orient='records')

def computeMap(df,index_df,model,fr="snippet",th=-200,n=1):
    cols = [c for c in list(df.columns) if c != 'Y']
    P = []
    Y = []
    len_m = None
    for sn in set(index_df[fr]):
        indices = list(index_df[index_df[fr]==sn].index)
        sp = df[df.index.isin(indices)]
        x_sp = sp[cols].values
        y_sp = np.array(sp['Y'])
        y = [i for i,v in enumerate(y_sp) if v == 1]
        Y.append(y)
        x_sp = xgb.DMatrix(x_sp, label=y_sp)
        #y_sp = xgb.DMatrix(np.array(y_sp))
        #pr = np.squeeze(model.predict(x_sp))
        try:
            pr = np.squeeze(model.predict_proba(x_sp))[:,1]
        except:
            pr = np.squeeze(model.predict(x_sp))
        p = list()
        for v in reversed(list(pr.argsort()[:])):
            if th is None:
                #print("uuuu")
                p.append(v)
            elif pr[v] > th:
                p.append(v)
        if len_m is None:
            len_m = len(p)
        elif len(p) < len_m:
            len_m = len(p)
        P.append(p)
    try:
        return metrics.mapk(
            Y,P,len_m
        )
    except:
        return -1

def computeAccBal(xy,y,model,th=0.8):
    #xp = np.squeeze(model.predict_classes(x))
    xp = [float(int(v >= 0.5)) for v in np.squeeze(model.predict(xy))]
    return balanced_accuracy_score(y, xp)


grid_search_param = {
    "n":[1,2,3],
    "features":["all","sim","un_int_card"],
    "prop":[1,0.8,0.6,0.5,0.3],
    "sampling":["under static","over"],
    "n":[1,2,3,4,5],
    "even":[True,False],
    "num_round":[10,50,100,200]
}


params_list = generateGridList(grid_search_param)

def fetNameV(v):
    if "cosine" in str(v):
        return "cosine_similarity"
    else:
        return v

try:
    done_keys = pickle.load(open(output_folder+"done_keys.p","rb"))
except:
    done_keys = set()

def filt_features(df,ff):
    if ff == "all":
        return df,list(df.columns)
    elif ff == "sim":
        feats = [col for col in df.columns if "sim" in col or "Y" == col]
        return df[feats],feats
    elif ff == "un_int_card":
        feats = [col for col in df.columns if "sim" not in col or "Y" == col]
        return df[feats],feats


def load_data(input_folder,ff):
    train_df,feats = filt_features(pd.read_csv(input_folder+"train.csv"),ff)
    train_index = pd.read_csv(input_folder+"train_index.csv")
    val_df,feats = filt_features(pd.read_csv(input_folder+"val.csv"),ff)
    val_index = pd.read_csv(input_folder+"val_index.csv")
    test_df,feats = filt_features(pd.read_csv(input_folder+"test.csv"),ff)
    test_index = pd.read_csv(input_folder+"test_index.csv")
    return train_df,train_index,val_df,val_index,test_df,test_index,feats

#import xgboost as xgb


def train(train_df,train_index,val_df,val_index,test_df,test_index,path,r,plst=plst):
    cols = [c for c in list(train_df.columns) if c != 'Y']



    path_scores = path + "score_obj.p"
    path_model = path + "model.h5"
    path_r = path + "r.p"
    X_train = train_df[cols].values
    Y_train = np.array(train_df['Y'])
    X_test = test_df[cols].values
    Y_test = np.array(test_df['Y'])
    X_val = val_df[cols].values
    Y_val = np.array(val_df['Y'])
    X = pd.concat([train_df,val_df])[cols].values
    Y = np.array(pd.concat([train_df,val_df])['Y'])
    if r["sampling"]=="under static":
        cols = [c for c in list(train_df.columns) if c != 'Y']
        training_generator = DataGenerator(train_df,train_index,n=r["n"],even=r["even"])
        X, Y = training_generator._generate()
        #X_train = train_df[cols].values
        #Y_train = np.array(train_df['Y'])
        #rus = RandomUnderSampler(sampling_strategy=1,random_state=42)
        #X_res, Y_res = rus.fit_resample(X_train, Y_train)
        #model.fit(x=X_res,y=Y_res, callbacks=[TestCallback(train_df, test_df, val_df, train_index, test_index, val_index, r["th"],r,output_folder+name+"/"),callback_es], validation_data=(X_val,Y_val), epochs=epochs)
        #rus = RandomUnderSampler(sampling_strategy=r["prop"],random_state=42)
        #X, Y = rus.fit_resample(X, Y)
    elif r["sampling"]=="over":
        ros = RandomOverSampler(sampling_strategy=r["prop"],random_state=42)
        X, Y = ros.fit_resample(X, Y)
    else:
        return 0

    dtrain = xgb.DMatrix(X, label=Y)
    #dtest = xgb.DMatrix(np.array(Y))

    XY_train = xgb.DMatrix(X_train, label=Y_train)
    #Y_train = xgb.DMatrix(np.array(Y_train))

    XY_test = xgb.DMatrix(X_test, label=Y_test)
    #Y_test = xgb.DMatrix(np.array(Y_test))

    XY_val = xgb.DMatrix(X_val, label=Y_val)
    #Y_val = xgb.DMatrix(np.array(Y_val))

    
    print("Training:",r)
    M = xgb.train( plst, dtrain, r['num_round'])
    #M.fit(X,Y)
    score_obj = {
        "accb_train":computeAccBal(XY_train,Y_train,M),
        "accb_test":computeAccBal(XY_test,Y_test,M),
        "Map_snippet_video_train":computeMap(train_df,train_index,M,fr="snippet"),
        "Map_video_snippet_train":computeMap(train_df,train_index,M,fr="video"),
        "Map_snippet_video_test":computeMap(test_df,test_index,M,fr="snippet"),
        "Map_video_snippet_test":computeMap(test_df,test_index,M,fr="video")
    }
    print("Scores",score_obj)
    os.makedirs(path)
    pickle.dump(r,open(path_r,'wb'))
    pickle.dump(M,open(path_model,'wb'))
    pickle.dump(score_obj,open(path_scores,"wb"))
    return 1

try:
    done_keys = pickle.load(open(output_folder+"done_keys.p","rb"))
except:
    done_keys = set()

for r in params_list:
    for feat in ["all","sim","un_int_card"]:
        r["algo"] = "XGBBoost"
        r_key = json.dumps({k:fetNameV(str(v)) for k,v in r.items()})
        if r_key not in done_keys:
            try:
                name = str(max([int(name) for name in os.listdir(output_folder) if os.path.isdir(output_folder+name)]) + 1)
            except:
                name = str(0)
            train_df,train_index,val_df,val_index,test_df,test_index,feats = load_data(input_folder,r["features"])
            path = output_folder+name+"/"
            #input(path)
            val = train(train_df,train_index,val_df,val_index,test_df,test_index,path,r)
            if val == 1:
                done_keys.add(r_key)
                pickle.dump(done_keys,open(output_folder+"done_keys.p","wb"))

            


            

