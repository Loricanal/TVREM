# TVREM: text to video and video to text retrieval for educational material

This repository contains the source code of a methodology that relies on named entities to automatically link text-based educational resources with educational videos.   
The methodology has been tested on two different datasets: 

* **BookToYout**: it consists of textbook paragraphs and youtube instructional videos. The youtube video transcripts and the texts extracted from PDF are available in the **tmp** folder

* **EDUCA**: it is formed by lecture notes and video lectures from some MIT courses.

The datasets can be downloaded at the following address: [https://kaggle.com/datasets/f35d99e30d00deea4a2162dacae2d665be7ab0ce18368ad97b7b6ce78933aab9](https://kaggle.com/datasets/f35d99e30d00deea4a2162dacae2d665be7ab0ce18368ad97b7b6ce78933aab9)

In the following we will describe Python scripts useful for re-running experiments or testing code on new data. 
A brief clarification on the meaning of the names used in the following is given here:
```
<dataset_name>: dataset name
<pdf_folder>: the path to the dataset folder containing the pdf files
<txt_folder>: the path to the dataset folder containing the plain txt files in case you don't have the pdf files (e.g. BookToYout dataset)
<video_folder>: the path to the dataset folder containing the MP4 files
<train_test_split>: the path to the json file containing the split between train, test and validation data
<temp_dir_txt>: the folder containing text files extracted from PDFs
<temp_dir_transcripts>: the folder containing the transcripts of the audio extracted from the video
<temp_dir_entities_txt>: the folder containing the entitites files derived from textual documents
<temp_dir_entities_video>: the folder containing the entitites files derived from video transcripts 
<temp_dir_entities_sets_txt>: the folder containing the entitites sets files derived from textual documents
<temp_dir_entities_sets_video>: the folder containing the entitites sets files derived from video
<temp_dir_features>: the folder containing the features files
<dir_models_nn>: the folder containing the neural netowrk models
<dir_models_algos>: the folder containing the models derived with scikit-learn library
<dir_models_bst>: the folder containing the xgboost models
```

## Converting PDF files to plain text 
```
python3 pdf_to_txt.py <pdf_folder> <temp_dir_txt>
```

## Extracting transcripts from videos
```
python3 from_video_to_transctipt.py <video_folder> <temp_dir_transcripts>
```

## Extracting entities derived from plain text files
```
python3 extract_entities.py <temp_dir_txt> <temp_dir_entities_txt>
```

## Extracting entities derived from video transcripts files
```
python3 extract_entities.py <temp_dir_transcripts> <temp_dir_entities_video>
```

## Enriching entities sets derived from text
```
python3 form_entity_sets.py <temp_dir_entities_txt> <temp_dir_entities_sets_txt>
```

## Enriching entities sets derived from video
```
python3 form_entity_sets.py <temp_dir_entities_video> <temp_dir_entities_sets_video>
```

## Creating features
```
python3 form_features.py <temp_dir_entities_sets_txt> <temp_dir_entities_sets_video> <train_test_split> <temp_dir_features>
```

## Train NN models
```
python3 train_nn.py <temp_dir_features>  <dir_models_nn>
```

## Train scikit learn algorithms models
```
python3 train_algos.py <temp_dir_features>  <dir_models_algos>
```

## Train scikit learn algorithms models
```
python3 train_bst.py <temp_dir_features>  <dir_models_bst>
```

## Evaluate 
```
python3 evaluate.py <temp_dir_features> <dir_models> <results_folder>
```

## Evaluate BERT+Cosine similarity ranking
```
python3 BERT.py  <temp_dir_features> <temp_dir_transcripts> <temp_dir_txt> <results_folder>
```

## Evaluate TFIDF+Cosine similarity ranking
```
python3 TokenSim.py <temp_dir_features> <temp_dir_transcripts> <temp_dir_txt> <results_folder>
```

