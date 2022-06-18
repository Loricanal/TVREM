import sys
import os
import json
import getopt
import os
import pickle
import sys
from functools import reduce
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
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

def jaccard_sim(A,B):
    try:
        return len(A&B)/len(A|B)
    except:
        return 0
    
def dice_sim(A,B):
    try:
        return 2*len(A&B)/(len(A)+len(B))
    except:
        return 0

def overlap_sim(A,B):
    try:
        return len(A&B)/min([len(A),len(B)])
    except:
        return 0
        

def norm_w_int_sim(A,B):
    try:
        return len(A&B)/max([len(A),len(B)])
    except:
        return 0

def cos_sim(A,B):
    try:
        return (len(A&B)**2)/(len(A)*len(B))
    except:
        return 0

def computeGraphsSims(A,B):
    return {
        "jaccard_sim":jaccard_sim(A,B),
        "overlap_sim":overlap_sim(A,B),
        "norm_w_int_sim":norm_w_int_sim(A,B),
        "int_card":len(A&B),
        "union_card":len(A|B)
    }


def createFeatures(o_ts,o_v):
    o = {
        "card_E_ts":len(o_ts["E"]),
        "card_E_v":len(o_v["E"]),
        "card_I_ts":len(o_ts["I"]),
        "card_I_v":len(o_v["I"]),
        "card_EP_ts":len(o_ts["EP"]),
        "card_EP_v":len(o_v["EP"])
    }
    for key in o_ts:
        v = computeGraphsSims(o_ts[key],o_v[key])
        for k in v:
            o[k+"___"+key] = v[k]
    return o



try:    
    input_folder_ts = sys.argv[1]
except:
    raise("You have to specify the input folder for snippets")
try:    
    input_folder_v = sys.argv[2]
except:
    raise("You have to specify the input folder for videos")
try:
    train_test_split = sys.argv[3]
except:
    raise("You have to specify the train test split")
try:
    gt_path = sys.argv[4]
except:
    raise("You have to specify gt")
try:    
    output_folder = sys.argv[5]
except:
    raise("You have to specify the output")


try:
    os.makedirs(output_folder)
except:
    pass

train_test=json.load(open(train_test_split,"r"))

gt = pd.read_csv(gt_path)

lf_train = list()
y_train = list()
train_index = list()
#input(gt)
for v1 in list(set(gt['snippet'])&set(train_test["train"])):
    for v2 in list(set(gt['video'])&set(train_test["train"])):
        f = createFeatures(pickle.load(open(input_folder_ts+v1+".p","rb")),pickle.load(open(input_folder_v+v2+".p","rb")))
        y = float(list(gt[(gt["snippet"] == v1) & (gt['video']==v2)]['value'])[0])
        y_train.append(y)
        train_index.append({"snippet":v1,"video":v2,"value":y})
        lf_train.append(f)

scaler = MinMaxScaler(feature_range=(-1,1))
train_df = pd.DataFrame(lf_train)
cols = list(train_df.columns)
#input(train_df)
train_values = train_df.values
#input(train_values)
new_values_train = scaler.fit_transform(train_values)
train_df = pd.DataFrame(new_values_train,columns=cols)
train_df['Y'] = y_train
#print(train_df)
train_df.to_csv(output_folder+"train.csv",index=False)

pd.DataFrame(train_index).to_csv(output_folder+"train_index.csv",index=False)

lf_test = list()
y_test = list()
test_index = list()

for v1 in list(set(gt['snippet'])&set(train_test["test"])):
    for v2 in list(set(gt['video'])&set(train_test["test"])):
        f = createFeatures(pickle.load(open(input_folder_ts+v1+".p","rb")),pickle.load(open(input_folder_v+v2+".p","rb")))
        y = float(list(gt[(gt["snippet"] == v1) & (gt['video']==v2)]['value'])[0])
        y_test.append(y)
        test_index.append({"snippet":v1,"video":v2,"value":y})
        lf_test.append(f)

test_df = pd.DataFrame(lf_test)
test_values = test_df.values
new_values_test = scaler.transform(test_values)
test_df = pd.DataFrame(new_values_test,columns=cols)
test_df['Y'] = y_test

test_df.to_csv(output_folder+"test.csv",index=False)
#print(test_df)
pd.DataFrame(test_index).to_csv(output_folder+"test_index.csv",index=False)

lf_val = list()
y_val = list()
val_index = list()
for v1 in list(set(gt['snippet'])&set(train_test["val"])):
    for v2 in list(set(gt['video'])&set(train_test["val"])):
        f = createFeatures(pickle.load(open(input_folder_ts+v1+".p","rb")),pickle.load(open(input_folder_v+v2+".p","rb")))
        y = float(list(gt[(gt["snippet"] == v1) & (gt['video']==v2)]['value'])[0])
        y_val.append(y)
        val_index.append({"snippet":v1,"video":v2,"value":y})
        lf_val.append(f)

val_df = pd.DataFrame(lf_val)
val_values = val_df.values
new_values_val = scaler.transform(val_values)
val_df = pd.DataFrame(new_values_val,columns=cols)
val_df['Y'] = y_val

val_df.to_csv(output_folder+"val.csv",index=False)
pd.DataFrame(val_index).to_csv(output_folder+"val_index.csv",index=False)




#if "._" != file[:2] and file.endswith(".pdf") and file.replace('.pdf','.txt') not in output_files