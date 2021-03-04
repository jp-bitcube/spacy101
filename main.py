import pandas as pd
import numpy as np
import spacy
from spacy.util import minibatch, compounding
from wordcloud import WordCloud, STOPWORDS
import re
import matplotlib.pyplot as plt
import random
# from sentiment_prediction.main import sentiment_predictions_with_tfid
# document1 = pd.read_table('data/drugsComTest_raw.tsv')
# document2 = pd.read_table('data/drugsComTrain_raw.tsv')
#
# concatDataSet = [document1, document2]
# keys = ['Test', 'Train']
# df = pd.concat(concatDataSet, keys=keys)
#
# df.to_csv("data/drug_review_dataset_with_sentiment.csv")

df = pd.read_csv('data/drug_review_dataset_with_sentiment.csv')

# NER
nlp = spacy.load('en_core_web_sm')
ner = nlp.get_pipe('ner')


def process_review(review):
    processed_token = []
    for token in review.split():
        token = ''.join(e.lower() for e in token if e.isalnum())
        processed_token.append(token)

    return ' '.join(processed_token)


drug_names = df['drugName'].unique().tolist()
drug_names = [x.lower() for x in drug_names]

reviews = df['review'].tolist()

count = 0
TRAINING_DATA = []
for _, item in df.iterrows():
    ent_dict = {}
    if count < 1000:
        review = process_review(item['review'])
        visited_items = []
        entities = []
        for token in review.split():
            if token in drug_names:
                for i in re.finditer(token, review):
                    if token not in visited_items:
                        entity = (i.span()[0], i.span()[1], 'DRUG')
                        visited_items.append(token)
                        entities.append(entity)
        if len(entities) > 0:
            ent_dict['entities'] = entities
            train_item = (review, ent_dict)
            TRAINING_DATA.append(train_item)
            count += 1

# print(TRAINING_DATA)

# sentiment_predictions_with_tfid(['Hello my name is Peter, I love eating'])