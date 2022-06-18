import pandas as pd

try:
    dataset = sys.argv[1]
except:
    raise("You have to specify the dataset")

results = pd.concat([pd.read_csv("results/DATASET_mymodels_results.csv"),
pd.read_csv("results/DATASET_algo_results_tokensim.csv"),
pd.read_csv("results/DATASET_algo_results_bert.csv"),
pd.read_csv("results/DATASET_algo_results_similarities.csv")
])

input(result.columns)

results.to_csv("results/EDUCA.csv",index=False)

