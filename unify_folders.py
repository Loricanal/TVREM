import sys
import os
import shutil

try:
    folders = ["tmp/"+sys.argv[1]+"/"+n for n in ["model_nn/","model_algos/","model_bst/"]] #"model_nn_sim/","model_algos_sim/","model_bst_sim/","model_nn_un_int_card/","model_algos_un_int_card/","model_bst_un_int_card/"
    folder_output = "tmp/"+sys.argv[1]+"/mymodels/"
except:
    raise("You have to specify dataset")

try:
    os.makedir(folder_output)
except:
    pass

for path in folders:
    #print(path.split("/")[-2])
    #input(path)
    dirs = [(path+f,f) for f in os.listdir(path) if os.path.isdir(path+f) and "_" not in f]
    #input(dirs)
    for d1,d2 in dirs:
        try:
            shutil.copytree(d1, folder_output+path.split("/")[-2]+"_"+d2)
        except:
            shutil.copytree(d1, folder_output+path.split("/")[-2]+"_"+d2)
            print("Failed",d1)



