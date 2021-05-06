# -*- coding: utf-8 -*-
"""
Created on Wed May  5 22:20:19 2021

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


df_train = pd.read_csv("./Corona_NLP_train.csv",error_bad_lines=False)


spacy.load('en_core_web_md')



df_train.columns




df_train.head()
print(df_train.shape)



#EDA- Nan values
var_null_train = [var for var in df_train.columns if df_train[var].isnull().sum() > 0]
print(var_null_train)
df_train[var_null_train].isnull().mean()



ax = df_train.Sentiment.value_counts().plot(kind='bar')
fig = ax.get_figure()



train_df = df_train[['OriginalTweet', 'Sentiment']]



train_df.shape



train_df.isnull().sum()



label_encoder = preprocessing.LabelEncoder()
train_df['Sentiment']= label_encoder.fit_transform(train_df['Sentiment'])
  
train_df['Sentiment'].unique()



train_df.head()


# ## tokenization of tweets using spacy's 'en_core_web_md' model




spacy_tok = spacy.load('en_core_web_md') #English Language model for tokenization!
sample_review = train_df.OriginalTweet[55]
sample_review



spacy_parsed_review = spacy_tok(sample_review)
spacy_parsed_review


# ## Using explacy to see the tokenization and POS tag


#!wget https://raw.githubusercontent.com/tylerneylon/explacy/master/explacy.py

url = 'https://raw.githubusercontent.com/tylerneylon/explacy/master/explacy.py'
filename = 'explacy.py'
urllib.request.urlretrieve(url, filename)



explacy.print_parse_info(spacy_tok, 'Covid-19 has taken many lives all over India and other nations')



explacy.print_parse_info(spacy_tok,train_df.OriginalTweet[1])


# ## Visualizing Lemma, POS , deposition , shape of an example


tokenized_text = pd.DataFrame()

for i, token in enumerate(spacy_parsed_review):
    tokenized_text.loc[i, 'text'] = token.text
    tokenized_text.loc[i, 'lemma'] = token.lemma_,
    tokenized_text.loc[i, 'pos'] = token.pos_
    tokenized_text.loc[i, 'tag'] = token.tag_
    tokenized_text.loc[i, 'dep'] = token.dep_
    tokenized_text.loc[i, 'shape'] = token.shape_
    tokenized_text.loc[i, 'is_alpha'] = token.is_alpha
    tokenized_text.loc[i, 'is_stop'] = token.is_stop
    tokenized_text.loc[i, 'is_punctuation'] = token.is_punct

tokenized_text[:20]



#spacy.displacy.render(spacy_parsed_review, style='ent', jupyter=True)
spacy.explain('npadvmod') # to explain POS tag



sentence_spans = list(spacy_parsed_review.sents)
sentence_spans



displacy.render(spacy_parsed_review, style='dep', jupyter=True,options={'distance': 140})




options = {'compact': True, 'bg': 'white','distance': 140,
           'color': 'blue', 'font': 'Trebuchet MS'}
displacy.render(spacy_parsed_review, jupyter=True, style='dep', options=options)



noun_chunks_df = pd.DataFrame()

for i, chunk in enumerate(spacy_parsed_review.noun_chunks):
    noun_chunks_df.loc[i, 'text'] = chunk.text
    noun_chunks_df.loc[i, 'root'] = chunk.root,
    noun_chunks_df.loc[i, 'root.text'] = chunk.root.text,
    noun_chunks_df.loc[i, 'root.dep_'] = chunk.root.dep_
    noun_chunks_df.loc[i, 'root.head.text'] = chunk.root.head.text

noun_chunks_df[:20]




nlp = spacy.load('en_core_web_md',disable=["tagger","ner"])



train_df['spacy_parsed'] = train_df.OriginalTweet.apply(nlp)
corpus = st.CorpusFromParsedDocuments(train_df,
                             category_col='Sentiment',
                             parsed_col='spacy_parsed').build()

from spacy.training import Example

#preparing data to feed to spacy pipeline i.e in form of list of tuples of text and label
train_df['tuples'] = train_df.apply(
    lambda row: (row['OriginalTweet'],row['Sentiment']), axis=1)
train = train_df['tuples'].tolist()

#functions from spacy documentation
def load_data(limit=0, split=0.8):
    train_data = train
    np.random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    cats = [{'POSITIVE': bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])

def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}

#("Number of texts to train from","t" , int)
n_texts=30000
#You can increase texts count if you have more computational power.

#("Number of training iterations", "n", int))
n_iter=10

# load the dataset
print("Loading Covid Tweets data...")
(train_texts, train_cats), (dev_texts, dev_cats) = load_data(limit=n_texts)
print("Using {} examples ({} training, {} evaluation)"
      .format(n_texts, len(train_texts), len(dev_texts)))
train_data = list(zip(train_texts,
                      [{'cats': cats} for cats in train_cats]))
#nlp = spacy.load('en_core_web_sm')  # create english Language class

'''
if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s'" % model)
   
else:
'''    
nlp = spacy.blank("en")  # create blank Language class
print("Created 'en' model")
    
# add the text classifier to the pipeline if it doesn't exist
# nlp.create_pipe works for built-ins that are registered with spaCy
if 'textcat' not in nlp.pipe_names:
        textcat = nlp.add_pipe('textcat')
       # nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
else:
        textcat = nlp.get_pipe('textcat')
    
# add label to text classifier
textcat.add_label('POSITIVE')

# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    #create Example object to pass in nlp.update (spacy v3 update)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example],
                           sgd=optimizer, drop=0.2,
                           losses=losses)
                    
            #print losses for every batch    
            #print("Losses", losses)
        with textcat.model.use_params(optimizer.averages):
        # evaluate on the dev data split off in load_data()
            scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))



"""
def save_model(output_dir, nlp, new_model_name):
    ''' This Function Saves model to 
    given output directory'''
    
    output_dir = f'../working/{output_dir}'
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

#function for loading data from spacy documentation
def load_data(limit=0, split=0.8):
    train_data = train
    #np.random.shuffle(train_data)
    train_data = train_data[-limit:] 
    texts, labels = zip(*train_data)
    cats = [{'POSITIVE': bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


#("Number of texts to train from","t" , int)
n_texts=30000

# load the dataset
print("Loading Covid Tweets data...")
(train_texts, train_cats), (dev_texts, dev_cats) = load_data(limit=n_texts)
print("Using {} examples ({} training, {} evaluation)"
      .format(n_texts, len(train_texts), len(dev_texts)))
train_data = list(zip(train_texts,
                      [{'cats': cats} for cats in train_cats]))

nlp = spacy.load('en_core_web_md')

#function for evaluating the model 
def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}


def train(train_data, output_dir, n_iter=10, model=None):
    
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s'" % model)
   
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created 'en' model")
    
    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.add_pipe('textcat')
       # nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe('textcat')
    
    # add label to text classifier
    textcat.add_label('POSITIVE')
    
    
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    #create Example object to pass in nlp.update (spacy v3 update)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example],
                           sgd=optimizer, drop=0.2,
                           losses=losses)
                    
            #print losses for every batch    
            #print("Losses", losses)
        with textcat.model.use_params(optimizer.averages):
        # evaluate on the dev data split off in load_data()
            scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))

        save_model(output_dir, nlp, 'textcat')

           


from spacy.training import Example
import os

model_path = 'models/sentiment_model'
train(train_data, model_path,n_iter=10, model=None)
"""


