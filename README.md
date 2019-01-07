# movie-dialogue-dev

This repository contains the code for NeurIPS 2018 paper "Towards Deep Conversational Recommendations" 
https://papers.nips.cc/paper/8180-towards-deep-conversational-recommendations

## Requirements

- Python 2.7
- PyTorch 0.4.1
- tqdm
- nltk
- h5py
- numpy
- scikit-learn

## Usage

### Get the data
- Get ReDial data from https://github.com/ReDialData/website/tree/data and unzip it.
Example of script to split `train_data.jsonl` into training and validation set
```python
import json, random
data = []
for line in open("train_data.jsonl"):
    data.append(json.loads(line))
random.shuffle(data)
n_data = len(data)
split_data = [data[:int(.8 * n_data)], data[int(.8 * n_data):]]

with open("train_data", 'w') as outfile:
    for example in split_data[0]:
        json.dump(example, outfile)
        outfile.write('\n')
with open("valid_data", 'w') as outfile:
    for example in split_data[1]:
        json.dump(example, outfile)
        outfile.write('\n')
```
- Get Movielens data https://grouplens.org/datasets/movielens/latest/. Note that for the paper we retrieved the Movielens
data set in September 2017. The Movielens latest dataset has been updated since then.
Example of script to split `ratings.csv` into training, validation and test sets
```python
import csv, random
data = []
with open("ratings.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != "userId":
            data.append(row)
            
random.shuffle(data)
n_data = len(data)
split_data = [data[:int(.8 * n_data)],
              data[int(.8 * n_data):int(.9 * n_data)],
              data[int(.9 * n_data):]]

with open("train_data", 'w') as outfile:
    writer = csv.writer(outfile)
    for dat in split_data[0]:
        writer.writerow(dat)
with open("valid_data", 'w') as outfile:
    writer = csv.writer(outfile)
    for dat in split_data[1]:
        writer.writerow(dat)
with open("test_data", 'w') as outfile:
    writer = csv.writer(outfile)
    for dat in split_data[2]:
        writer.writerow(dat)
```
- Merge the movie lists by matching the movie names from ReDial and Movielens. Note that this will create an intermediate file `movies_matched.csv`, which is deleted at the end of the script.
```
python match_movies.py --redial_movies_path=/path/to/redial/movies_with_mentions.csv --ml_movies_path=/path/to/movielens/movies.csv --destination=/path/to/redial/merged_movie_list.csv
```

### Specify the paths

In the `config.py` file, specify the different paths to use:

- Model weights will be saved in folder `MODELS_PATH='/path/to/models`
- ReDial data in folder `REDIAL_DATA_PATH=/path/to/redial'`.
This folder must contain three files called `train_data`, `valid_data` and `test_data`
- Movielens data in folder `ML_DATA_PATH='/path/to/movielens'`.
This folder must contain three files called `train_ratings`, `valid_ratings` and `test_ratings`

### Get GenSen pre-trained models

Get GenSen pre-trained models from https://github.com/Maluuba/gensen.
More precisely, you will need the embeddings in the `/path/to/models/embeddings` folder, and 
the following model files: `nli_large_vocab.pkl`, `nli_large.model` in the `/path/to/models/GenSen` folder
```
wget https://genseniclr2018.blob.core.windows.net/models/nli_large_vocab.pkl
wget https://genseniclr2018.blob.core.windows.net/models/nli_large.model
```

### Train models

- Train sentiment analysis. This will train a model to predict the movie form labels from ReDial.
The model will be saved in the `/path/to/models/sentiment_analysis` folder
```
python train_sentiment_analysis.py
```
- Train autoencoder recommender system. This will pre-train an Autoencoder Recommender system on Movielens, then fine-tune it on ReDial.
The model will be saved in the `/path/to/models/autorec` folder 
```
python train_autorec.py
```
- Train conversational recommendation model. This will train the whole conversational recommendation model, using the previously trained models.
 The model will be saved in the `/path/to/models/recommender` folder.
```
python train_recommender.py
```

### Generate sentences
`generate_responses.py` loads a trained model. 
It takes real dialogues from the ReDial dataset and lets the model generate responses whenever the human recommender speaks
(responses are conditioned on the current dialogue history).
```
python generate_responses.py --model_path=/path/to/models/recommender/model_best --save_path=/path/to/save/generations
```
