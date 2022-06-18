#load libraries

from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf
import itertools
import pandas as pd
import ml_metrics as metrics
import pickle
import sys
import keras
from keras import Sequential
import keras.backend as K
from keras.layers import Dense, Input
from keras.layers.core import Dropout
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.regularizers import l2
from sklearn.metrics import f1_score as f1_score_func
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn import preprocessing
import numpy as np
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
import xgboost as xgb


try:
    features_folder = sys.argv[1]
except:
    raise("You have to specify the features folder")
try:
    models_folder = sys.argv[2]
except:
    raise("You have to specify the features folder")
try:
    output_folder = sys.argv[3]
except:
    raise("You have to specify the features folder")


try:

    os.makedirs("results/")
except:
    pass
input_folder = features_folder
folder = models_folder
output = output_folder+"mymodels_results.csv"
output1 = output_folder+"DATASET_mymodels_results_complete.csv"
results = list()



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

#print(input_folder)
#input()


folders_models = [folder]


def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles



def openKeras(f):
    #print(f)
    #m = keras.models.load_model(f)
    try:
        m = keras.models.load_model(f)
        return m
    except:
        try:
            m = pickle.load(open(f,"rb"))
            return m
        except:
            return False


from IPython.display import display, HTML
pd.set_option('display.max_rows', None)



def getPredictionsforMAP(df,index_df,model,fr="snippet",n=1):
    cols = [c for c in list(df.columns) if c != 'Y']
    P = []
    Y = []
    len_m = None
    if fr == "video":
        to = "snippet"
    elif fr == "snippet":
        to = "video"
    records = list()
    #print(index_df[(index_df[fr]=="96b9e69b81cc6229728c9206cca9f72d")&(index_df[to]=="96b9e69b81cc6229728c9206cca9f72d")])
    for sn in set(index_df[fr]):
        #print("Sn",sn)
        #display(index_df[index_df[fr]==sn])
        indices = list(index_df[index_df[fr]==sn].index)
        #print("indices",indices)
        sp = df[df.index.isin(indices)]
        x_sp = sp[cols].values
        y_sp = np.array(sp['Y'])
        y = [i for i,v in enumerate(y_sp) if v == 1]
        Y.append(y)
        #input(type(model))
        #input(len(model.feature_names))
        #input(len(cols))
        #input(len(model.feature_names))
        try:
            pr = np.squeeze(model.predict_proba(x_sp))[:,1]
        except:
            try:
                #print(1)
                pr = np.squeeze(model.predict_proba(x_sp))
            #print(3)
            except:
                try:
                    pr = np.squeeze(model.predict(x_sp))
                except:
                    #input(x_sp)
                    tr = sp[cols]
                    tr.columns = ["f"+str(i) for i in range(len(cols))]
                    pr = np.squeeze(model.predict(xgb.DMatrix(tr, label=y_sp)))
        #print(sorted(pr))
        p = list(reversed(list(pr.argsort()[:]))) #[::-1]
        #print(y)
        #input(p)
        pr_ord = [pr[v] for v in p]
        if len_m is None:
            len_m = len(p)
        elif len(p) < len_m:
            len_m = len(p)
        ranked_list = ".."
        ranked_list_to = ",".join(list(index_df[to]))
        ranked_list_y = ""
        records.append({
            fr:sn,
            to+"_yid":ranked_list_y,
            to+"_pid":ranked_list,
            to+"_pid_unord":",".join(list(index_df[to])),
            to+'_y':",".join([str(y)]),
            to+'_p':",".join([str(v) for v in p]),
            to+'_p_unord':",".join(np.array(index_df.index).astype(str)),
            to+'_prob':",".join([str(v) for v in pr]),
            to+'_probord':",".join([str(v) for v in pr_ord])
            
        })
        P.append(p)
    return Y,P,len_m,pd.DataFrame(records)


def computeMap(Y,P,th=10):
    return metrics.mapk(
        Y,P,th
    )



#Recall= (Relevant_Items_Recommended in top-k) / (Relevant_Items)

#Precision= (Relevant_Items_Recommended in top-k) / (k_Items_Recommended)


def precisionAtK(Y,P,K):  
    pm = list()
    for i,y in enumerate(Y):
        p = P[i][:K]
        pm.append(len(set(p) & set(y)) / K)
    return np.mean(pm)

def recallAtK(Y,P,K):
    pm = list()
    for i,y in enumerate(Y):
        if len(set(y)) > 0:
            p = P[i][:K]
            pm.append(len(set(p) & set(y)) / len(set(y)))
    return np.mean(pm)


def computeAccBal(x,y,model,th=0.5):
    xp = [float(int(v >= th)) for v in np.squeeze(model.predict(x))]
    return balanced_accuracy_score(y, xp)


def precision_recall_f1_acc(x,y,model,th=0.5):
    xp_1 = [float(int(v >= th)) for v in np.squeeze(model.predict(x))]
    xp_0 = [float(int(v < th)) for v in np.squeeze(model.predict(x))]
    y_1 = y 
    y_0 = [float(int(v<0.7)) for v in y]
    o = dict()
    o['Pr_1_'+str(th)] = precision_score(y_1, xp_1)
    o['Rc_1_'+str(th)] = recall_score(y_1, xp_1)
    o['F1_1_'+str(th)] = f1_score(y_1, xp_1)
    o['Acc_1_'+str(th)] = accuracy_score(y_1, xp_1)
    o['Pr_0_'+str(th)] = precision_score(y_0, xp_0)
    o['Rc_0_'+str(th)] = recall_score(y_0, xp_0)
    o['F1_0_'+str(th)] = f1_score(y_0, xp_0)
    o['Acc_0_'+str(th)] = accuracy_score(y_0, xp_0)
    return o


def computeScores(model,X_train, Y_train, X_val, Y_val, X_test, Y_test, train_df,train_index, val_df,val_index, test_df,test_index):
    predictions = dict()
    
    Ytrain_sv,Ptrain_sv,len_mtrain_sv,predictions["snippet_video_train"] = getPredictionsforMAP(train_df,train_index,model,fr="snippet")
    Ytrain_vs,Ptrain_vs,len_mtrain_vs,predictions["video_snippet_train"] = getPredictionsforMAP(train_df,train_index,model,fr="video")
    
    #print(Ytrain_vs,"\n",Ptrain_vs,"\n",len_mtrain_vs)
    #input()
    
    Ytest_sv,Ptest_sv,len_mtest_sv,predictions["snippet_video_test"] = getPredictionsforMAP(test_df,test_index,model,fr="snippet")
    Ytest_vs,Ptest_vs,len_mtest_vs,predictions["video_snippet_test"] = getPredictionsforMAP(test_df,test_index,model,fr="video")
    
    Yval_sv,Pval_sv,len_mval_sv,predictions["snippet_video_val"] = getPredictionsforMAP(val_df,val_index,model,fr="snippet")
    Yval_vs,Pval_vs,len_mval_vs,predictions["video_snippet_val"] = getPredictionsforMAP(val_df,val_index,model,fr="video")
    
    #computeMap(Y,P,len_m)
    score_obj = {
        "Map_snippet_video_train":computeMap(Ytrain_sv,Ptrain_sv,len_mtrain_sv),
        "Map_video_snippet_train":computeMap(Ytrain_vs,Ptrain_vs,len_mtrain_vs),
        "Map_snippet_video_test":computeMap(Ytest_sv,Ptest_sv,len_mtest_sv),
        "Map_video_snippet_test":computeMap(Ytest_vs,Ptest_vs,len_mtest_vs),
        "Map_snippet_video_val":computeMap(Yval_sv,Pval_sv,len_mval_sv),
        "Map_video_snippet_val":computeMap(Yval_vs,Pval_vs,len_mval_vs),
        "Map1_snippet_video_train":computeMap(Ytrain_sv,Ptrain_sv,1),
        "Map1_video_snippet_train":computeMap(Ytrain_vs,Ptrain_vs,1),
        "Map1_snippet_video_test":computeMap(Ytest_sv,Ptest_sv,1),
        "Map1_video_snippet_test":computeMap(Ytest_vs,Ptest_vs,1),
        "Map1_snippet_video_val":computeMap(Yval_sv,Pval_sv,1),
        "Map1_video_snippet_val":computeMap(Yval_vs,Pval_vs,1),
        "Map5_snippet_video_train":computeMap(Ytrain_sv,Ptrain_sv,5),
        "Map5_video_snippet_train":computeMap(Ytrain_vs,Ptrain_vs,5),
        "Map5_snippet_video_test":computeMap(Ytest_sv,Ptest_sv,5),
        "Map5_video_snippet_test":computeMap(Ytest_vs,Ptest_vs,5),
        "Map5_snippet_video_val":computeMap(Yval_sv,Pval_sv,5),
        "Map5_video_snippet_val":computeMap(Yval_vs,Pval_vs,5),
        "Map10_snippet_video_train":computeMap(Ytrain_sv,Ptrain_sv,10),
        "Map10_video_snippet_train":computeMap(Ytrain_vs,Ptrain_vs,10),
        "Map10_snippet_video_test":computeMap(Ytest_sv,Ptest_sv,10),
        "Map10_video_snippet_test":computeMap(Ytest_vs,Ptest_vs,10),
        "Map10_snippet_video_val":computeMap(Yval_sv,Pval_sv,10),
        "Map10_video_snippet_val":computeMap(Yval_vs,Pval_vs,10),
        "P1_snippet_video_train":precisionAtK(Ytrain_sv,Ptrain_sv,1),
        "P1_video_snippet_train":precisionAtK(Ytrain_vs,Ptrain_vs,1),
        "P1_snippet_video_test":precisionAtK(Ytest_sv,Ptest_sv,1),
        "P1_video_snippet_test":precisionAtK(Ytest_vs,Ptest_vs,1),
        "P1_snippet_video_val":precisionAtK(Yval_sv,Pval_sv,1),
        "P1_video_snippet_val":precisionAtK(Yval_vs,Pval_vs,1),
        "P5_snippet_video_train":precisionAtK(Ytrain_sv,Ptrain_sv,5),
        "P5_video_snippet_train":precisionAtK(Ytrain_vs,Ptrain_vs,5),
        "P5_snippet_video_test":precisionAtK(Ytest_sv,Ptest_sv,5),
        "P5_video_snippet_test":precisionAtK(Ytest_vs,Ptest_vs,5),
        "P5_snippet_video_val":precisionAtK(Yval_sv,Pval_sv,5),
        "P5_video_snippet_val":precisionAtK(Yval_vs,Pval_vs,5),    
        "P10_snippet_video_train":precisionAtK(Ytrain_sv,Ptrain_sv,10),
        "P10_video_snippet_train":precisionAtK(Ytrain_vs,Ptrain_vs,10),
        "P10_snippet_video_test":precisionAtK(Ytest_sv,Ptest_sv,10),
        "P10_video_snippet_test":precisionAtK(Ytest_vs,Ptest_vs,10),
        "P10_snippet_video_val":precisionAtK(Yval_sv,Pval_sv,10),
        "P10_video_snippet_val":precisionAtK(Yval_vs,Pval_vs,10),  
        "R1_snippet_video_train":recallAtK(Ytrain_sv,Ptrain_sv,1),
        "R1_video_snippet_train":recallAtK(Ytrain_vs,Ptrain_vs,1),
        "R1_snippet_video_test":recallAtK(Ytest_sv,Ptest_sv,1),
        "R1_video_snippet_test":recallAtK(Ytest_vs,Ptest_vs,1),
        "R1_snippet_video_val":recallAtK(Yval_sv,Pval_sv,1),
        "R1_video_snippet_val":recallAtK(Yval_vs,Pval_vs,1),
        "R5_snippet_video_train":recallAtK(Ytrain_sv,Ptrain_sv,5),
        "R5_video_snippet_train":recallAtK(Ytrain_vs,Ptrain_vs,5),
        "R5_snippet_video_test":recallAtK(Ytest_sv,Ptest_sv,5),
        "R5_video_snippet_test":recallAtK(Ytest_vs,Ptest_vs,5),
        "R5_snippet_video_val":recallAtK(Yval_sv,Pval_sv,5),
        "R5_video_snippet_val":recallAtK(Yval_vs,Pval_vs,5),    
        "R10_snippet_video_train":recallAtK(Ytrain_sv,Ptrain_sv,10),
        "R10_video_snippet_train":recallAtK(Ytrain_vs,Ptrain_vs,10),
        "R10_snippet_video_test":recallAtK(Ytest_sv,Ptest_sv,10),
        "R10_video_snippet_test":recallAtK(Ytest_vs,Ptest_vs,10),
        "R10_snippet_video_val":recallAtK(Yval_sv,Pval_sv,10),
        "R10_video_snippet_val":recallAtK(Yval_vs,Pval_vs,10) 
    }
    return score_obj


count = 0

try:
    results = pickle.load(open("results_"+dataset+".p","rb"))
    done_files = pickle.load(open("done_files_"+dataset+".p","rb"))
except:
    done_files = set()
    results = list()

for fold in folders_models:

    #settings_files = {f:pickle.load(open(f,"rb")) for f in getListOfFiles(fold) if "r.p" in f}
    models_files = [f for f in getListOfFiles(fold) if "model.h5" in f]

    tot = len(models_files)
    count = 0
    for f in models_files:
        count += 1
        if f not in done_files:
            print("File:",f,"\n","Count:",count,"on",tot)
            model = openKeras(f)
            #input(1)
            if model != False:
                try:
                    settings = pickle.load(open(f.replace("model.h5","r.p"),"rb"))
                except:
                    settings = model.get_params()
                    settings['algo'] = str(type(model).__name__)
                    if model.n_features_in_ == 26:
                        settings['features'] = "all"
                    elif model.n_features_in_ == 12:
                        settings['features'] = "sim"
                    else:
                        settings['features'] = "un_int_card"
                    #input(settings['algo'])
                #print(f)
                train_df,train_index,val_df,val_index,test_df,test_index,feats = load_data(input_folder,settings["features"])
                cols = [c for c in list(train_df.columns) if c != 'Y']
                x = train_df[cols].values
                try:
                    y = np.array(train_df['Y'])
                except:
                    print(train_df.columns)
                    raise
                X_train, Y_train, X_test, Y_test, X_val, Y_val = train_df[cols].values, np.array(train_df['Y']), test_df[cols].values, np.array(test_df['Y']), val_df[cols].values, np.array(val_df['Y'])
                #input(settings)
                #result_obj = {"id":f,"algorithm":"NN","parameters":str(settings)}
                #input(2)
                #input(settings)
                score_obj = computeScores(model,X_train, Y_train, X_val, Y_val, X_test, Y_test, train_df,train_index, val_df,val_index, test_df,test_index)
                #input(3)
                if "algo" in settings:
                    result_obj = {"id":f,"algorithm":settings["algo"],"parameters":str(settings)}
                else:
                    result_obj = {"id":f,"algorithm":"NN","parameters":str(settings)}
                #for k in settings:
                    #result_obj[k] = settings[k]
                #input(4)
                print(result_obj)
                for k in score_obj:
                    result_obj[k] = score_obj[k]
                #input(5)
                results.append(result_obj)
                done_files.add(f)
                print(result_obj)
                pickle.dump(results,open("results_"+dataset+".p","wb"))

                pickle.dump(done_files,open("done_files_"+dataset+".p","wb"))
            else:
                print("Failed",f)

                
df = pd.DataFrame(results)
records =list()
for v in df.to_dict(orient="records"):
    mean_val = np.mean([v["Map1_snippet_video_val"],v["Map1_video_snippet_val"]])
    v['Map1_mean'] = mean_val
    records.append(v)


df = pd.DataFrame(records).sort_values(by='Map1_mean', ascending=False)


df.to_csv(
    "results/"+output1,
    index=False
)

records =list()
algos = set()
for v in df.to_dict(orient="records"):
    if v['algorithm'] not in algos:
        records.append(v)
        algos.add(v['algorithm'])

df = pd.DataFrame(records)
df.to_csv(
    "results/"+output,
    index=False
)


pickle.dump({"best_model":list(df['id'])[0]},open("results/best_model_"+dataset+".p","wb"))



