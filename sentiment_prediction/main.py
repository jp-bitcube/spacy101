import pandas as pd

import string

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


punctuation = string.punctuation
nlp = spacy.load('en_core_web_sm')

# http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
imbd_doc = pd.read_table('data/imdb_labelled.txt')
amazon_doc = pd.read_table('data/amazon_cells_labelled.txt')
yelp_doc = pd.read_table('data/yelp_labelled.txt')

concatDataSet = [imbd_doc, amazon_doc, yelp_doc]

# Giving the Datasets headers
for colname in concatDataSet:
    colname.columns = ['Message', 'Target']

# Adding Keys for each dataset
keys = ['Yelp', 'IMDB', 'Amazon']

# df = pd.concat(concatDataSet, keys=keys)

# # Shape of dataset
# print(f"Shape = {df.shape}")
#
# # Head of the table
# print(f"Head of table = {df.head()}")
#
# # Data to CSV
# df.to_csv("data/sentiment_dataset1.csv")
df = pd.read_csv('data/sentiment_dataset1.csv')
# Data Cleaning
# print(f"Columns = {df.columns}")
# print(f"Missing Values = {df.isnull().sum()}")

# Build a list of stopwords to use to filter
stopwords = list(STOP_WORDS)
# print(stopwords)

# Testing the nlp or the spacy_tokens function
# docX = "This is how John Walker was walking. He was also running beside the lawn"

# for word in docX:
#     print(word.text, " -- ", word.pos_, " -- ", word.lemma_)

# for word in docX:
#     if word.pos_ != 'PRON':
#         print(word.lemma_.lower().strip())

# List Comprehensions of our Lemma
# comp = [word.lemma_.lower().strip() if word.pos_ != 'PRON' else word.lower_ for word in docX]
# removedPunc = [word for word in docX if word.is_stop == False and not word.is_punct]


def spacy_tokens(sentence):
    my_tokens = nlp(sentence)
    my_tokens = [token.lemma_.lower().strip() if token.pos_ != 'PRON' else token.lower_ for token in my_tokens]
    my_tokens = [word for word in my_tokens if word not in stopwords and word not in punctuation]
    return my_tokens


# Custom transformer using spaCy
class Predictors(TransformerMixin):

    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fir_params):
        return self

    def get_params(self, deep=True):
        return {}


# Basic function to clean text
def clean_text(text):
    return text.strip().lower()


# Vectorization
vector = CountVectorizer(tokenizer=spacy_tokens, ngram_range=(1, 1))
classifier = LinearSVC()

# Splitting DataSet
X = df['Message']
y_labels = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)

# Create the pipeline to clean, tokenize, vectorized, and classify
pipe = Pipeline([
    ("cleaner", Predictors()),
    ("vector", vector),
    ("classifier", classifier)
])
# Using Tfid
# Create the pipeline to clean, tokenize, vectorized, and classify
tf_vector = TfidfVectorizer(tokenizer=spacy_tokens)
pipe_tfid = Pipeline([
    ("cleaner", Predictors()),
    ("vector", tf_vector),
    ("classifier", classifier)
])

# Predictions Results
# 1 = Positive Review
# 0 = Negative Review
# for (sample, prediction) in zip(X_test, sample_prediction):
#     print(sample, "Prediction => ", prediction)

# # Another '3' reviews
example = [
    "I do enjoy my job",
    "What a poor product!, I will have to get a new one",
    "I feel amazing!"
]


def sentiment_predictions_with_tfid(prediction_array):
    # Fit our data
    data = pipe_tfid.fit(X_train, y_train)
    print(data)
    # Predicting wih a test dataset
    sample_prediction = pipe_tfid.predict(X_test)
    print(sample_prediction)
    # Accuracy
    print("Accuracy: ", pipe_tfid.score(X_test, y_test))
    print("Accuracy: ", pipe_tfid.score(X_train, y_train))
    print("Accuracy: ", pipe_tfid.score(X_test, sample_prediction))
    print(pipe_tfid.predict(prediction_array))


def sentiment_predictions_with_count_vector(prediction_array):
    data = pipe.fit(X_train, y_train)
    print(data)
    # Predicting wih a test dataset
    sample_prediction = pipe.predict(X_test)
    print(sample_prediction)
    # Accuracy
    print("Accuracy: ", pipe.score(X_test, y_test))
    print("Accuracy: ", pipe.score(X_train, y_train))
    print("Accuracy: ", pipe.score(X_test, sample_prediction))
    print(pipe.predict(prediction_array))
