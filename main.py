# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:34:13 2021

@author: sahil
"""

import pandas as pd
import numpy as np
import spacy
from spacy import displacy
from spacy.util import minibatch, compounding
from sklearn import preprocessing


import matplotlib.pyplot as plt


df_train = pd.read_csv('./Corona_NLP_train.csv')

df=pd.DataFrame(data = {'Name': ['ff','gg','hh','yy'],
                        'Age':[24,12,48,30]},
                dtype = np.int32)
                 
                 
  # Import label encoder

#label_encoder = preprocessing.LabelEncoder()
#df_train['Sentiment']= label_encoder.fit_transform(df_train['Sentiment'])
  
#df_train['Sentiment'].unique()


  
"""
Given an array of ints, return True if .. 1, 2, 3, .. appears in the array somewhere. 
array123([1, 1, 2, 3, 1]) → True
array123([1, 1, 2, 4, 1]) → False
array123([1, 1, 2, 1, 2, 3]) → True

def array123(nums):
    for i in range(len(nums)):
        
        if nums[i] == 1 and nums[i+1] == 2 and nums[i+2] == 3:
            return True
    return False


a = 'xxccaazz'
b='xxbaaz'

if len(a) < 2 or len(b) < 0:
    print(0)
    
else:
    c = []
    for i in range(len(a)-1):
        c.append(a[i:i+2])  
        
        occurence = list(map(lambda x: x in b, c))
        #print(occurence)
    print(occurence.count(True))
"""

