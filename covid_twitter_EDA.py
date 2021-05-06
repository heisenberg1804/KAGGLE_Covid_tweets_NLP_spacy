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


df_train = pd.read_csv('./Corona_NLP_train.csv')



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
                             parsed_col='parsed').build()





