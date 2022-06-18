from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf
import itertools
import pandas as pd
import ml_metrics as metrics
import pickle
import sys
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
from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler
import random

try:
    input_folder = sys.argv[1]
except:
    raise("You have to specify the features folder")
try:
    input_folder_v = sys.argv[2]
except:
    raise("You have to specify the video transcripts folder")
try:
    input_folder_s = sys.argv[3]
except:
    raise("You have to specify the texts folder")
try:
    output_folder = sys.argv[4]
except:
    raise("You have to specify the output folder")


try:
    os.makedirs("results/")
except:
    pass

def generateGridList(my_dict):
    keys, values = zip(*my_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return pd.DataFrame(permutations_dicts).drop_duplicates().to_dict(orient='records')

grid_search_param = {
    "n_features":[20,50,100,200,500,1000],
    "ngram_range":[(1,1),(2,2),(3,3),(1,3),(1,2),(2,3)],
    "max_df":[0.7,0.8,0.9,1.0],
    "min_df":[1,3,5,7,10]
}

import os 
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


input_files_v = [input_folder_v+f for f in os.listdir(input_folder_v) if '.txt' in f or '.xml' in f and '._' not in f]
input_files_s = [input_folder_s+f for f in os.listdir(input_folder_s) if '.txt' in f or '.xml' in f and '._' not in f]

text_v = {f.split("/")[-1].split(".")[0]:open(f).read() for f in input_files_v}
text_s = {f.split("/")[-1].split(".")[0]:open(f).read() for f in input_files_s}

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def fetNameV(v):
    if "cosine" in str(v):
        return "cosine_similarity"
    else:
        return v

params_list = generateGridList(grid_search_param)




try:
    models_names = list()
    text_v_vect = pickle.load(open("text_v_vect_DATASET_tokensim.p".replace("DATASET",dataset),"rb"))
    text_s_vect = pickle.load(open("text_s_vect_DATASET_tokensim.p".replace("DATASET",dataset),"rb"))
    #text_v_vect = pickle.load(open("text_v_vect_tokensim.p","rb"))
    #text_s_vect = pickle.load(open("text_s_vect_tokensim.p","rb"))
    for r in params_list:
        O = {k:fetNameV(str(v)) for k,v in r.items()}
        O["algo"] = "Tfidf"
        r_key = json.dumps(O)
        models_names.append(r_key)
    print("Done")
except:
    text_v_vect = dict()
    text_s_vect = dict()
    models_names = list()
    for r in params_list:
        O = {k:fetNameV(str(v)) for k,v in r.items()}
        O["algo"] = "Tfidf"
        r_key = json.dumps(O)
        
        models_names.append(r_key)
        
        v_keys = list(sorted(text_v.keys()))
        corpus_v = [text_v[k] for k in v_keys]
        tfidf_v = TfidfVectorizer(stop_words=stop_words,
                                  max_df=r["max_df"],
                                  min_df=r["min_df"],
                                  ngram_range=r["ngram_range"],
                                  max_features=r["n_features"]
                                 ).fit_transform(corpus_v).toarray()
        text_v_vect_p = {(r_key,k):tfidf_v[i] for i,k in enumerate(v_keys)}
        
        s_keys = list(sorted(text_s.keys()))
        corpus_s = [text_s[k] for k in s_keys]
        tfidf_s = TfidfVectorizer(stop_words=stop_words,
                                  max_df=r["max_df"],
                                  min_df=r["min_df"],
                                  ngram_range=r["ngram_range"],
                                  max_features=r["n_features"]
                                 ).fit_transform(corpus_s).toarray()
        text_s_vect_p = {(r_key,k):tfidf_s[i] for i,k in enumerate(s_keys)}
        
        for k in text_v_vect_p:
            text_v_vect[k] = text_v_vect_p[k]
            
        for k in text_s_vect_p:
            text_s_vect[k] = text_s_vect_p[k]
    pickle.dump(text_v_vect,open("text_v_vect_DATASET_tokensim.p".replace("DATASET",dataset),"wb"))
    pickle.dump(text_s_vect,open("text_s_vect_DATASET_tokensim.p".replace("DATASET",dataset),"wb"))


import ml_metrics as metrics
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

train_index = pd.read_csv(input_folder+"train_index.csv")
val_index = pd.read_csv(input_folder+"val_index.csv")
test_index = pd.read_csv(input_folder+"test_index.csv")



def computeMap(Y,P,th=10):
    return metrics.mapk(
        Y,P,th
    )

def precisionAtK(Y,P,K):  
    pm = list()
    for i,y in enumerate(Y):
        p = P[i][:K]
        pm.append(len(set(p) & set(y)) / K)
    return np.mean(pm)

def recallAtK(Y,P,K):
    pm = list()
    for i,y in enumerate(Y):
        p = P[i][:K]
        pm.append(len(set(p) & set(y)) / len(set(y)))
    return np.mean(pm)    
    
def getPredictionsforMAP(index_df,model,fr="snippet",n=1):
    global text_v_vect,text_s_vect
    P = []
    Y = []
    if fr == "snippet":
        to = "video"
    elif fr == "video":
        to = "snippet"
    for sn in set(index_df[fr]):
        df = index_df[index_df[fr]==sn].reset_index(drop=True)
        y = list(df[df[to]==sn].index)[0]
        pr = np.array([util.cos_sim(text_v_vect[(model,r['video'])],text_s_vect[(model,r['snippet'])]).numpy()[0][0] for r in df.to_dict(orient="records")])
        Y.append([y])
        p = list(reversed(list(pr.argsort()[:])))
        len_m = len(p)
        P.append(p)
    return Y,P,len_m


scores = list()
for model in models_names:
    Ytrain_sv,Ptrain_sv,len_mtrain_sv = getPredictionsforMAP(train_index,model,fr="snippet")
    Ytrain_vs,Ptrain_vs,len_mtrain_vs = getPredictionsforMAP(train_index,model,fr="video")
    Ytest_sv,Ptest_sv,len_mtest_sv = getPredictionsforMAP(test_index,model,fr="snippet")
    Ytest_vs,Ptest_vs,len_mtest_vs = getPredictionsforMAP(test_index,model,fr="video")
    Yval_sv,Pval_sv,len_mval_sv = getPredictionsforMAP(val_index,model,fr="snippet")
    Yval_vs,Pval_vs,len_mval_vs = getPredictionsforMAP(val_index,model,fr="video")
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

    result_obj = {"id":0,"algorithm":"TokenSim","parameters":model}

    for k in score_obj:
        result_obj[k] = score_obj[k]

    scores.append(result_obj)


    
SCORES = pd.DataFrame(scores)

records=list()
for v in SCORES.to_dict(orient="records"):
    mean_val = np.mean([v["Map1_snippet_video_val"],v["Map1_video_snippet_val"]])
    v['Map1_mean'] = mean_val
    records.append(v)


df = pd.DataFrame(records).sort_values(by='Map1_mean', ascending=False).head(1)


df.to_csv(
    output_folder+"algo_results_tokensim.csv",
    index=False
)

