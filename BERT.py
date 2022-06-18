import os 
from sentence_transformers import SentenceTransformer

try:
    dataset = sys.argv[1]
except:
    raise("You have to specify the dataset")

try:
    os.makedirs("results/")
except:
    pass
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



input_files_v = [input_folder_v+f for f in os.listdir(input_folder_v) if '.txt' in f or '.xml' in f and '._' not in f]
input_files_s = [input_folder_s+f for f in os.listdir(input_folder_s) if '.txt' in f or '.xml' in f and '._' not in f]
text_v = {f.split("/")[-1].split(".")[0]:open(f).read() for f in input_files_v}
text_s = {f.split("/")[-1].split(".")[0]:open(f).read() for f in input_files_s}

models_names = ['bert-base-nli-mean-tokens', 
 'bert-large-nli-mean-tokens', 
 'roberta-base-nli-mean-tokens',
 'roberta-large-nli-mean-tokens',
 'distilbert-base-nli-mean-tokens',
 'bert-base-nli-stsb-mean-tokens',
 'bert-large-nli-stsb-mean-tokens',
 'roberta-base-nli-stsb-mean-tokens',
 'roberta-large-nli-stsb-mean-tokens',
 'distilbert-base-nli-stsb-mean-tokens',
 'distiluse-base-multilingual-cased',
 'xlm-r-base-en-ko-nli-ststb',
 'xlm-r-large-en-ko-nli-ststb',
 'paraphrase-MiniLM-L6-v2']

import pickle
try:
    text_v_vect = pickle.load(open("text_v_vect_DATASET_bert.p".replace("DATASET",dataset),"rb"))
    text_s_vect = pickle.load(open("text_s_vect_DATASET_bert.p".replace("DATASET",dataset),"rb"))
except:
    text_v_vect = dict()
    text_s_vect = dict()

    for m in models_names:
        model = SentenceTransformer(m)
        print("Model:",m)
        print("Len:",len(text_v))
        for i,k in enumerate(list(text_v.keys())):
            if i % 10 == 0:
                print("I",i)
                pickle.dump(text_v_vect,open("text_v_vect_DATASET_bert.p".replace("DATASET",dataset),"wb"))
            key = (m,k)
            if key not in text_v_vect:
                embeddings = model.encode([text_v[k]])
                text_v_vect[key] = embeddings[0]
                
        print("Len:",len(text_s))
        for i,k in enumerate(list(text_s.keys())):
            if i % 10 == 0:
                print("I",i)
                pickle.dump(text_s_vect,open("text_s_vect_DATASET_bert.p".replace("DATASET",dataset),"wb"))
            key = (m,k)
            if key not in text_s_vect:
                embeddings = model.encode([text_s[k]])
                text_s_vect[key] = embeddings[0]

    pickle.dump(text_v_vect,open("text_v_vect_DATASET_bert.p".replace("DATASET",dataset),"wb"))
    pickle.dump(text_s_vect,open("text_s_vect_DATASET_bert.p".replace("DATASET",dataset),"wb"))


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
    result_obj = {"id":0,"algorithm":"BERT "+model,"parameters":""}
    for k in score_obj:
        result_obj[k] = score_obj[k]
    scores.append(result_obj)

    
SCORES = pd.DataFrame(scores)

records=list()
for v in SCORES.to_dict(orient="records"):
    mean_val = np.mean([v["Map1_snippet_video_val"],v["Map1_video_snippet_val"]])
    v['Map1_mean'] = mean_val
    records.append(v)

pd.DataFrame(records).sort_values(by='Map1_mean', ascending=False).to_csv(
    output_folder+"algo_results_bert.csv",
    index=False
)
