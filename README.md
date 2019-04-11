# movie-dialogue-dev

This repository contains the code for NeurIPS 2018 paper "Towards Deep Conversational Recommendations" 
https://arxiv.org/abs/1812.07617

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
Get ReDial data from https://github.com/ReDialData/website/tree/data and Movielens data https://grouplens.org/datasets/movielens/latest/. Note that for the paper we retrieved the Movielens
data set in September 2017. The Movielens latest dataset has been updated since then.
```
git clone https://github.com/RaymondLi0/conversational-recommendations.git
cd conversational-recommendations
pip install -r requirements.txt
python -m nltk.downloader punkt

mkdir -p redial movielens
wget -O redial/redial_dataset.zip https://github.com/ReDialData/website/raw/data/redial_dataset.zip
wget -O movielens/ml-latest.zip http://files.grouplens.org/datasets/movielens/ml-latest.zip
# split ReDial data
python scripts/split-redial.py redial/
mv redial/test_data.jsonl redial/test_data
# split Movielens data
python scripts/split-movielens.py movielens/
```

Merge the movie lists by matching the movie names from ReDial and Movielens. Note that this will create an intermediate file `movies_matched.csv`, which is deleted at the end of the script.
```
python scripts/match_movies.py --redial_movies_path=redial/movies_with_mentions.csv --ml_movies_path=movielens/ml-latest/movies.csv --destination=redial/movies_merged.csv
```

### Specify the paths

In the `config.py` file, specify the different paths to use:

- Model weights will be saved in folder `MODELS_PATH='/path/to/models'`
- ReDial data in folder `REDIAL_DATA_PATH='/path/to/redial'`.
This folder must contain three files called `train_data`, `valid_data` and `test_data`
- Movielens data in folder `ML_DATA_PATH='/path/to/movielens'`.
This folder must contain three files called `train_ratings`, `valid_ratings` and `test_ratings`

### Get GenSen pre-trained models

Get GenSen pre-trained models from https://github.com/Maluuba/gensen.
More precisely, you will need the embeddings in the `/path/to/models/embeddings` folder, and 
the following model files: `nli_large_vocab.pkl`, `nli_large.model` in the `/path/to/models/GenSen` folder
```
cd /path/to/models
mkdir GenSen embeddings
wget -O GenSen/nli_large_vocab.pkl https://genseniclr2018.blob.core.windows.net/models/nli_large_vocab.pkl
wget -O GenSen/nli_large.model https://genseniclr2018.blob.core.windows.net/models/nli_large.model
cd embeddings
wget https://raw.githubusercontent.com/Maluuba/gensen/master/data/embedding/glove2h5.py
wget https://github.com/Maluuba/gensen/raw/master/data/embedding/glove2h5.sh
sh glove2h5.sh
cd /path/to/project_dir
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
python generate_responses.py --model_path=/path/to/models/recommender/model_best --save_path=generations
```
