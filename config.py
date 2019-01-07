import os

path = os.path.dirname(os.path.basename(__file__))

# models path
MODELS_PATH = '/path/to/models'
AUTOREC_MODEL = os.path.join(MODELS_PATH, "autorec")
SENTIMENT_ANALYSIS_MODEL = os.path.join(MODELS_PATH, 'sentiment_analysis')
RECOMMENDER_MODEL = os.path.join(MODELS_PATH, "recommender")
# data set path
REDIAL_DATA_PATH = '/path/to/redial'
TRAIN_PATH = "train_data"
VALID_PATH = "valid_data"
TEST_PATH = "test_data"

MOVIE_PATH = os.path.join(REDIAL_DATA_PATH, "movies_merged.csv")
VOCAB_PATH = os.path.join(REDIAL_DATA_PATH, "vocabulary.p")

# reddit data path
REDDIT_PATH = "/path/to/fb_movie_dialog_dataset/task4_reddit"
REDDIT_TRAIN_PATH = "task4_reddit_train.txt"
REDDIT_VALID_PATH = "task4_reddit_dev.txt"
REDDIT_TEST_PATH = "task4_reddit_test.txt"

CONVERSATION_LENGTH_LIMIT = 40  # conversations are truncated after 40 utterances
UTTERANCE_LENGTH_LIMIT = 80  # utterances are truncated after 80 words

# Movielens ratings path
ML_DATA_PATH = "/path/to/movielens"
ML_SPLIT_PATHS = [os.path.join(ML_DATA_PATH, "split0"),
                  os.path.join(ML_DATA_PATH, "split1"),
                  os.path.join(ML_DATA_PATH, "split2"),
                  os.path.join(ML_DATA_PATH, "split3"),
                  os.path.join(ML_DATA_PATH, "split4"), ]
ML_TRAIN_PATH = "train_ratings"
ML_VALID_PATH = "valid_ratings"
ML_TEST_PATH = "test_ratings"