# -*- coding: utf-8 -*-
"""
Created on Fri May  7 00:40:03 2021

@author: sahil
"""

from __future__ import unicode_literals, print_function
from __future__ import unicode_literals

from pathlib import Path

import pandas as pd
import spacy
import copy
from spacy.util import minibatch, compounding
import re
from spacy.training import Example


def clean_string(mystring):
    return re.sub('[^A-Za-z\ 0-9 ]+', '', mystring)



def train(model=None, output_dir=None, n_iter=10):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.add_pipe('textcat')

    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe('textcat')

    # add label to text classifier
    for i in ['Neutral', 'Positive', 'Extremely Negative', 'Extremely Positive']:
        textcat.add_label(i)


    df = pd.read_csv('Corona_NLP_train.csv', error_bad_lines=False)
    df.drop(['Location', 'ScreenName', 'TweetAt', 'UserName', ], axis=1, inplace=True)
    #df = df[df['Sentiment'] != 'empty']

    sentiment_values = df['Sentiment'].unique()
    labels_default = dict((v, 0) for v in sentiment_values)

    train_data = []
    for i, row in df.iterrows():

        label_values = copy.deepcopy(labels_default)
        label_values[row['Sentiment']] = 1

        train_data.append((str(clean_string(row['OriginalTweet'])), {"cats": label_values}))

    train_data = train_data[:5000]

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        print('{:^5}\t'.format('LOSS'))
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
                print("Losses", losses)
            #print('{0:.3f}'  # print a simple table
                  #.format(losses['textcat']))

    # test the trained model
    test_text = "Modi government is so bad"
    doc = nlp(test_text)
    print(test_text, sorted(doc.cats.items(), key=lambda val: val[1], reverse=True))

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)



