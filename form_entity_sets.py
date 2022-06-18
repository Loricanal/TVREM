from utils.request import *
import pickle
import sys
import os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

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

fold = '/'.join(output_folder.split("/")[:-2])+'/wikidata/'

try:
    os.makedirs(fold)
except:
    pass

classof_path = fold + "classof.p"
subclassof_path = fold + "subclassof.p"
partof_path = fold + "partof.p"

try:
    classof,subclassof,partof = pickle.load(open(classof_path)), pickle.load(open(subclassof_path)), pickle.load(open(partof_path))
except:
    classof,subclassof,partof = dict(),dict(),dict()

def getClass(wd_uri):
    global classof
    try:
        return classof[wd_uri]
    except:
        print("Getting class for:",wd_uri)
        endpoint = 'https://query.wikidata.org/sparql'
        query = Q_WD_INSTANCE.replace('STR_TO_SUB',wd_uri)
        print("Query:",query)
        df = getSPARQLResponse(query,endpoint)
        if len(df):
            cols = list(df.columns)
            classof[wd_uri] = set(df[cols[-1]])
            return set(df[cols[-1]])
        classof[wd_uri] = set()
        return set()

def getSubClass(wd_uri):
    global subclassof
    try:
        return subclassof[wd_uri]
    except:
        print("Getting sub class for:",wd_uri)
        endpoint = 'https://query.wikidata.org/sparql'
        query = Q_WD_SUBCLASS.replace('STR_TO_SUB',wd_uri)
        print("Query:",query)
        df = getSPARQLResponse(query,endpoint)
        if len(df):
            cols = list(df.columns)
            subclassof[wd_uri] = set(df[cols[-1]])
            return set(df[cols[-1]])
        subclassof[wd_uri] = set()
        return set()

def getPartOf(wd_uri):
    global partof
    try:
        return partof[wd_uri]
    except:
        print("Getting part of for:",wd_uri)
        endpoint = 'https://query.wikidata.org/sparql'
        query = Q_WD_SUBCLASS.replace('STR_TO_SUB',wd_uri)
        print("Query:",query)
        df = getSPARQLResponse(query,endpoint)
        if len(df):
            cols = list(df.columns)
            partof[wd_uri] = set(df[cols[-1]])
            return set(df[cols[-1]])
        partof[wd_uri] = set()
        return set()

def parseEntities(f):
    print("File",f)
    o = json.load(open(f))['entities']
    entities = set()
    for ext in o:
        for item in o[ext]:
            if 'wikidataUri' in item and item['wikidataUri']:
                entities.add(item['wikidataUri'])
    return entities

def createSets(E,max_err=20):
    sets = {
        "E":E,
        "I":set(),
        "P":set()
    }
    E_l = list(E)
    for i,e in enumerate(E_l):
        print(i,"of",len(E_l))
        flag = True
        error_count = 0
        while flag:
            if error_count > max_err:
                raise
            try:
                cl = getClass(e)
                if cl:
                    sets["I"].add(e)
                    for c in cl:
                        sets["P"].add(c)
                sb = getSubClass(e)
                for c in sb:
                    sets["P"].add(c)
                pt = getPartOf(e)
                for c in pt:
                    sets["P"].add(c)
                flag = False
            except:
                error_count += 1
    sets["EP"] = sets["E"] | sets["P"]
    return sets

input_files = [f for f in os.listdir(input_folder) if '.json' in f and '._' not in f[:2]]

output_files = [file for file in os.listdir(output_folder)]

print("Output files len:",len(output_files))

#input()

todo_files = [file for file in os.listdir(input_folder) if "._" != file.split('/')[-1][:2] and file.split('/')[-1].replace('.json','.p') not in output_files]

print("Todo files:",len(todo_files))

for f in todo_files:
    print("File:",f)
    path = input_folder+f
    #blockPrint()
    entities_f = parseEntities(path)
    sets = createSets(entities_f)
    pickle.dump(classof,open(classof_path,"wb"))
    pickle.dump(subclassof,open(subclassof_path,"wb"))
    pickle.dump(partof,open(partof_path,"wb"))
    pickle.dump(sets,open(output_folder+f.replace('.json','.p'),"wb"))
    #enablePrint()
