# -*- coding: utf-8 -*-
"""
Created on Thu May  6 23:43:27 2021

@author: sahil
"""


import pandas as pd
import numpy as np
import spacy
from spacy import displacy
from spacy.util import minibatch, compounding
from sklearn import preprocessing

import explacy
import urllib.request
import scattertext as st


import matplotlib.pyplot as plt
from spacy.training import Example

df_train = pd.read_csv("./Corona_NLP_train.csv",error_bad_lines=False)

df_train = df_train[['OriginalTweet']]

#encoding sentiments
#label_encoder = preprocessing.LabelEncoder()
#df_train['Sentiment']= label_encoder.fit_transform(df_train['Sentiment'])

#visualizing frequency for each sentiment
ax=df_train.Sentiment.value_counts().plot(kind='bar')
fig = ax.get_figure()
fig.show("sentiment")

#df_train.Sentiment[df_train.sentiment<=3]=0
#df_train.Score[df_train.sentiment>=4]=1

nlp = spacy.load('en_core_web_md',disable=["tagger","ner"])

#preparing data to feed to spacy pipeline i.e in form of list of tuples of text and label
df_train['tuples'] = df_train.apply(
    lambda row: (row['OriginalTweet'],row['Sentiment']), axis=1)
train_df = df_train['tuples'].tolist()

#adding textCategorizer pipeline
textcat = nlp.add_pipe('textcat',last=True)


#load data function
import random

def load_data(limit=0, split=0.8):
    train_data=train_df
    # Shuffle the data
    random.shuffle(train_data)
    texts, labels = zip(*train_data)
    # get the categories for each review
    cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y), "NEUTRAL": y} for y in labels]

    # Splitting the training and evaluation data
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])

n_texts=23486

# Calling the load_data() function 
(train_texts, train_cats), (dev_texts, dev_cats) = load_data(limit=n_texts)

# Processing the final format of training data
train_data = list(zip(train_texts,[{'cats': cats} for cats in train_cats]))
train_data[:10]

















