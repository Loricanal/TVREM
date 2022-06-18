from utils.extractors import dandelion,dbspotlight,babelfy,textrazor,meaning_cloud,opencalais,dbspotlight,fox,spacyner,adel,google
from utils.request import *
from utils.tokenization import *
import sys
import glob
import os
import json
import shutil
from bs4 import BeautifulSoup
from time import sleep
import hashlib,pickle

credentials_apis = json.load(open('credentials.json'))

credential_index = {key:0 for key in credentials_apis}


try:    
    input_folder = sys.argv[1]
except:
    raise("You have to specify the input folder")
try:
    output_folder = sys.argv[2]
except:
    raise("You have to specify the output folder")


try:
    os.makedirs("tmp/failed_resp/")
except:
    pass


EXTRACTORS = [
    babelfy.BABELFY(credentials_apis['babelfy'][0]),
    textrazor.TEXTRAZOR(credentials_apis['textrazor'][0]),
    google.GOOGLE(credentials_apis['google'][0])
]

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def splitText(text,n):
    total_len = len(text)
    split_n = round(total_len/n)
    occs = findOccurrences(text,'\n')
    texts = []
    spls = [0]
    for j in range(n-1):
        spl = split_n*(j+1)
        for o in occs:
            if o > spl:
                spl = o+1
                break
        texts.append(text[spls[j]:spl])
        spls.append(spl)
    texts.append(text[spls[-1]:])
    return texts

def getEntities(text,extractors=EXTRACTORS,lang='en',credentials_apis=credentials_apis,credential_index=credential_index):
    ensemble_response = {'text':text,'entities':{}}
    for ext in extractors:
        flag = True
        count = 0
        fl = True
        while flag:
            try:
                ext.extract(text,lang=lang)
                ext.parse()
                ext_entities = ext.get_annotations()
                if type(ext_entities) == dict:
                    for key in ext_entities:
                        ensemble_response['entities'][ext.name+'___'+key] = ext_entities[key]
                else:
                    ensemble_response['entities'][ext.name] = ext_entities
                flag = False
            except Exception as e:
                print("ERROR:",e)
                if "key expired" in str(e):
                    ind = credentials_apis[ext.name].index(ext.api_key)
                    if ind == len(credentials_apis[ext.name])-1:
                        ind = 0
                    else:
                        ind += 1
                    ext.change_credendials(credentials_apis[ext.name][ind])
                elif "Bad Gateway" in str(e):
                    raise("Extractor:",ext.name)
                count += 1
                if count > 6:
                    fl = False
                    flag = False
                    if not "Large" in str(e) and not "Bad Request" in str(e):
                        raise("Extractor:",ext.name)
                    n = 2
                    while "Large" in str(e) or "Bad Request" in str(e):
                        texts = splitText(text,n)
                        try:
                            exts_t = []
                            for t in texts:
                                ext.extract(t,lang=lang)
                                ext.parse()
                                ext_entities = ext.get_annotations()
                                exts_t+=ext_entities
                            ensemble_response['entities'][ext.name] = exts_t
                            fl = True
                            break
                        except Exception as e2:
                            e = e2
                            if "Large" in str(e2):
                                n += 2
                            elif "Bad Request" in str(e2):
                                n += 2
                            else:
                                raise
                    if not fl:
                        erro = {
                            "err":str(e),
                            "text":text,
                            "ext_name":ext.name
                        }
                        hash_object = hashlib.md5((ext.name+text).encode())
                        pickle.dump(erro,open("failed_resp/"+str(hash_object.hexdigest())+".json","wb"))
    return ensemble_response

try:
    os.makedirs(output_folder)
except:
    pass

input_files = [f for f in os.listdir(input_folder) if '.txt' in f or '.xml' in f and '._' not in f]

output_files = [f for f in os.listdir(output_folder) if '.json' in f]

print("Input files:",len(input_files))
print("Output files:",len(output_files))

import queue
import threading
import time

#num_worker_threads = 4
num_worker_threads = 4

def worker():
    global output_files,input_folder,input_files
    while True:
        f = q.get()
        if f is None:
            break
        #print("File n:",str(i+1)+"/"+str(len(input_files)))
        print("Len queue:",q.qsize())
        text_f = open(input_folder+f).read()
        while True:
            try:
                if '.txt' in f:
                    f_json = f.replace('.txt','.json')
                elif '.xml' in f:
                    soup = BeautifulSoup(text_f,features="lxml")
                    text_f = soup.find('text').string
                    f_json = f.replace('.xml','.json')
                if f_json not in output_files:
                    #print("File:",f)
                    global TEXT
                    TEXT = text_f
                    entities = getEntities(text_f)
                    json.dump(entities,open(output_folder+f_json,"w"))
                    
                else:
                    exctractors_dict = {
                        "babelfy":babelfy.BABELFY(credentials_apis['babelfy'][0]),
                        "textrazor":textrazor.TEXTRAZOR(credentials_apis['textrazor'][0]),
                        "google":textrazor.TEXTRAZOR(credentials_apis['google'][0])

                    }
                    res = json.load(open(output_folder+f_json,"r"))
                    entities = res['entities']
                    for k in list(exctractors_dict.keys()):
                        if k in entities:
                            del exctractors_dict[k]
                    if len(exctractors_dict):
                        print("Missing:",f,list(exctractors_dict.keys()))
                        res2 = getEntities(text_f,extractors=list(exctractors_dict.values()))
                        ent2 = res2['entities']
                        flg = False
                        for k2 in ent2:
                            if k2 not in entities:
                                flg = True
                                entities[k2] = ent2[k2]
                        if flg:
                            res['entities'] = entities
                            json.dump(res,open(output_folder+f_json,"w"))
                    else:
                        print("Already done:",f)
                q.task_done()
                break
            except Exception as e:
                print("BBB",e,f)
                #time.sleep(5)
                q.put(f)
                break

        


#for item in input_files:
    #if '._' not in item:
        #worker(item)


q = queue.Queue()

threads = []

for i in range(num_worker_threads):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)


for item in input_files:
    if '._' not in item:
        q.put(item)

# block until all tasks are done
q.join()

print('stopping workers!')

# stop workers
for i in range(num_worker_threads):
    q.put(None)

for t in threads:
    t.join()
