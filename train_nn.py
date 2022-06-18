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

def computeMap(df,index_df,model,fr="snippet",n=1):
    #input(th)
    cols = [c for c in list(df.columns) if c != 'Y']
    P = []
    Y = []
    len_m = None
    for sn in set(index_df[fr]):
        indices = list(index_df[index_df[fr]==sn].index)
        sp = df[df.index.isin(indices)]
        x_sp = sp[cols].values
        y_sp = np.array(sp['Y'])
        #y = np.argmax(y_sp)
        #input(y)
        #input(np.max(y_sp))
        #input(y_sp)
        y = [i for i,v in enumerate(y_sp) if v == 1]
        #input(y)
        Y.append(y)
        pr = np.squeeze(model.predict_proba(x_sp))
        p = list(reversed(list(pr.argsort()[:]))) #[::-1]
        #for v in reversed(list(pr.argsort()[:][::-1])):
            #if th is None:
                #print("uuuu")
                #p.append(v)
            #elif pr[v] > th:
                #p.append(v)
        if len_m is None:
            len_m = len(p)
        elif len(p) < len_m:
            len_m = len(p)
        P.append(p)
    
    return metrics.mapk(
        Y,P,len_m
    )

def computeAccBal(x,y,model,th=0.8):
    #xp = np.squeeze(model.predict_classes(x))
    xp = [float(int(v >= 0.5)) for v in np.squeeze(model.predict(x))]
    return balanced_accuracy_score(y, xp)

class TestCallback(Callback):
    def __init__(self,train_df, test_df, val_df, train_index, test_index, val_index, th,r,output_folder):
        cols = [c for c in list(train_df.columns) if c != 'Y']
        self.output_folder = output_folder
        self.path_scores = output_folder + "score_obj.p"
        self.path_hist = output_folder + "history.p"
        self.path_model = output_folder + "model.h5"
        self.path_r = output_folder + "r.p"
        try:
            self.history_scores = pickle.load(open(self.path_hist,"rb"))
            #r_old = [pickle.load(open(self.path_r,"rb"))]
        except:
            self.history_scores = list()
        try:
            sco = pickle.load(open(self.path_scores,"rb"))
            self.val_map_avg = np.mean([sco["Map_snippet_video_val"],sco["Map_video_snippet_val"]])
            self.train_map_avg = np.mean([sco["Map_snippet_video_train"],sco["Map_video_snippet_train"]])
        except:
            self.val_map_avg = 0
            self.train_map_avg = 0
        self.train_df = train_df
        self.X_train = train_df[cols].values
        self.Y_train = np.array(train_df['Y'])
        self.test_df = test_df
        self.X_test = test_df[cols].values
        self.Y_test = np.array(test_df['Y'])
        self.X_val = val_df[cols].values
        self.Y_val = np.array(val_df['Y'])
        self.val_df = val_df
        self.th = th
        self.r = r
        #print("R:",self.r )

    def on_epoch_end(self, epoch, logs={}):
        score_obj = {
            "accb_train":computeAccBal(self.X_train, self.Y_train, self.model,self.th),
            "accb_test":computeAccBal(self.X_test, self.Y_test, self.model,self.th),
            "accb_val":computeAccBal(self.X_val, self.Y_val, self.model,self.th),
            "train_loss":model.evaluate(self.X_train, self.Y_train)[0],
            "val_loss":model.evaluate(self.X_val, self.Y_val)[0],
            "Map_snippet_video_train":computeMap(self.train_df,train_index,model,fr="snippet",n=3),
            "Map_video_snippet_train":computeMap(self.train_df,train_index,model,fr="video",n=3),
            "Map_snippet_video_test":computeMap(self.test_df,test_index,model,fr="snippet",n=3),
            "Map_video_snippet_test":computeMap(self.test_df,test_index,model,fr="video",n=3),
            "Map_snippet_video_val":computeMap(self.val_df,val_index,model,fr="snippet",n=3),
            "Map_video_snippet_val":computeMap(self.val_df,val_index,model,fr="video",n=3),
            "epoch":epoch
        }
        print("R:",self.r)
        print("Scores:",score_obj)
        if len(self.history_scores) == 0:
            self.history_scores.append(score_obj)
            try:
                os.makedirs(self.output_folder)
                #print("Folder created")
            except:
                pass
            self.model.save(self.path_model)
            pickle.dump(score_obj,open(self.path_scores,"wb"))
            pickle.dump(self.history_scores,open(self.path_hist,"wb"))
            pickle.dump(self.r,open(self.path_r,"wb"))
        else:
            mean_val = np.mean([score_obj["Map_snippet_video_val"],score_obj["Map_video_snippet_val"]])
            mean_train = np.mean([score_obj["Map_snippet_video_train"],score_obj["Map_video_snippet_train"]])
            #print("Mean val:",mean_val)
            #print("Val Map Avg:",self.val_map_avg)
            #print("Mean train:",mean_train)
            #print("Train Map Avg:",self.train_map_avg)
            if mean_val>=self.val_map_avg and mean_train>=self.train_map_avg:
                #print("SAVED")
                try:
                    os.makedirs(self.output_folder)
                    #print("Folder created")
                except:
                    pass
                self.history_scores.append(score_obj)
                self.model.save(self.path_model)
                pickle.dump(score_obj,open(self.path_scores,"wb"))
                pickle.dump(self.history_scores,open(self.path_hist,"wb"))
                pickle.dump(self.r,open(self.path_r,"wb"))
                self.train_map_avg = mean_train
                self.val_map_avg = mean_val
                """
                old_scores = self.history_scores[-1]
                sum_new = 0
                sum_old = 0
                for k in score_obj.keys():
                    if "Map" in k:
                        sum_new += score_obj[k]
                        sum_old += old_scores[k]
                if sum_new > sum_old:
                    self.history_scores.append(score_obj)
                    self.model.save(self.path_model)
                    pickle.dump(score_obj,open(self.path_scores,"wb"))
                    pickle.dump(self.history_scores,open(self.path_hist,"wb"))
                    pickle.dump(self.r,open(self.path_r,"wb"))
                """

grid_search_param = {
    "activation_middle":["linear","selu","elu","relu","sigmoid"],
    "loss_function":[
        'mse',
        tf.keras.losses.cosine_similarity,
        'binary_crossentropy'
    ],
    "dropout":[0.0,0.05, 0.10],
    #"batch":[10, 50, 100, 250, 500],
    "units":[(15,7),(10,5),(30,15)],
    "n":[1,2,3,4],
    "even":[True,False],
    "prop":[1,0.8,0.6,0.5],
    "sampling":["under","over","under static"],
    #"th":[-200,0.2,0.4,0.6,0.8],
    "th":[-200],
    "features":["all","sim","un_int_card"],
    "activation":["linear","sigmoid"]
}

reg_alpha = 0.000
epochs = 500
#units = 100
eg_alpha = 0.0
optimizer = 'adam'
activation = 'sigmoid'
patience = 5
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

for i,r in enumerate(params_list):
    if r["sampling"]=="under":
        del r['prop']
    elif r["sampling"]=="over":
        del r['even'],r['n']
    elif r["sampling"]=="under static":
        del r['prop']
    r_key = json.dumps({k:fetNameV(str(v)) for k,v in r.items()})
    if r_key not in done_keys:
        print(i,"of",len(params_list))
        try:
            name = str(max([int(name) for name in os.listdir(output_folder) if os.path.isdir(output_folder+name)]) + 1)
        except:
            name = str(0)
        train_df,train_index,val_df,val_index,test_df,test_index,feats = load_data(input_folder,r["features"])
        model = Sequential()
        model.add(Dense(r["units"][0], activation=r["activation_middle"], kernel_regularizer=l2(reg_alpha),
                        bias_regularizer=l2(reg_alpha), input_dim=len(train_df.columns)-1))
        model.add(Dropout(r["dropout"]))
        model.add(Dense(r["units"][1], activation=r["activation_middle"], kernel_regularizer=l2(reg_alpha),
                        bias_regularizer=l2(reg_alpha)))
        model.add(Dropout(r["dropout"]))
        model.add(Dense(1, activation=r['activation'], kernel_regularizer=l2(reg_alpha), bias_regularizer=l2(reg_alpha)))
        model.compile(loss=r["loss_function"], optimizer=optimizer, metrics=['mae', 'accuracy'])
        callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        print(model.summary())
        fla = 0
        cols = [c for c in list(train_df.columns) if c != 'Y']
        X_val = val_df[cols].values
        Y_val = np.array(val_df['Y'])

        #input(train_df)

        #input(train_index)
        if r["sampling"]=="under":
            fla = 1
            training_generator = DataGenerator(train_df,train_index,n=r["n"],even=r["even"])
            model.fit_generator(generator=training_generator,
                                use_multiprocessing=True,
                                callbacks=[TestCallback(train_df, test_df, val_df, train_index, test_index, val_index, r["th"],r,output_folder+name+"/"),callback_es],
                                epochs=epochs,
                                validation_data=(X_val,Y_val),
                                workers=4)

        elif r["sampling"]=="under static":
            fla = 1
            cols = [c for c in list(train_df.columns) if c != 'Y']
            training_generator = DataGenerator(train_df,train_index,n=r["n"],even=r["even"])
            X_res, Y_res = training_generator._generate()
            #X_train = train_df[cols].values
            #Y_train = np.array(train_df['Y'])
            #rus = RandomUnderSampler(sampling_strategy=1,random_state=42)
            #X_res, Y_res = rus.fit_resample(X_train, Y_train)
            model.fit(x=X_res,y=Y_res, callbacks=[TestCallback(train_df, test_df, val_df, train_index, test_index, val_index, r["th"],r,output_folder+name+"/"),callback_es],
                                validation_data=(X_val,Y_val),
                                epochs=epochs)

        elif r["sampling"]=="over":
            fla = 1
            cols = [c for c in list(train_df.columns) if c != 'Y']
            X_train = train_df[cols].values
            Y_train = np.array(train_df['Y'])
            ros = RandomOverSampler(sampling_strategy=r["prop"],random_state=42)
            X_res, Y_res = ros.fit_resample(X_train, Y_train)
            model.fit(x=X_train,y=Y_train,
                                callbacks=[TestCallback(train_df, test_df, val_df, train_index, test_index, val_index, r["th"],r,output_folder+name+"/"),callback_es],
                                validation_data=(X_val,Y_val),
                                epochs=epochs
                                )
        if fla == 1:
            done_keys.add(r_key)
            pickle.dump(done_keys,open(output_folder+"done_keys.p","wb"))
    else:
        print("Already done:",r_key)

    

















