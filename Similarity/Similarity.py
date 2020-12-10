#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:41:25 2020

@author: nora
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
from IPython.display import display




units_df=pd.read_excel("KFUPM-Units_NEW.xlsx", header=0)
available_departments=units_df["id"].str.strip()
Titles=units_df["All titles"].str.strip()

num_depts = len(available_departments)



#deptids = ["dept_" + str(i) for i in range(num_depts)]

# create a dictionary
dept_dict = dict(zip(available_departments, Titles))

# get all the dept ids in a list
ids = list(dept_dict.keys())


# create all possible pairs
pairs = []
# create a list of tuples
for i, v in enumerate(ids):
    for j in ids[i+1:]:
        pairs.append((ids[i], j))
        
print("There are a total of " + str(len(pairs)) + " pairs")
print("Displaying first 10 pairs: ")
display(pairs[:10])
print("....")
print("Displaying last 10 pairs: ")
display(pairs[-10:])


from normalization import *

# cleanTextdepts takes a list of strings and returns a list of lists
corpus = cleanTextdepts(Titles)

# convert list of lists into a list of strings
norm_dept_corpus = [' '.join(text) for text in corpus]

# display normalized corpus
display(norm_dept_corpus)

vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1,1))

'''
TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.float64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=0.0,
        ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,
        stop_words=None, strip_accents=None, sublinear_tf=False,
        token_pattern='(?u)\\b\\w\\w+\\b', tokenizer=None, use_idf=True,
        vocabulary=None)
'''
# calculate the feature matrix
feature_matrix = vectorizer.fit_transform(norm_dept_corpus).astype(float)

# display the shape of feature matrix 
display(feature_matrix.shape)

# display the first feature vector
display(feature_matrix[0])

# display the dense version of the feature vector
display(feature_matrix.toarray()[0])

# display the shape of dense feature vector
display(feature_matrix.toarray()[0].shape)

# display the first document text
display(norm_dept_corpus[0])


def compute_cosine_similarity(pair):
    
    # extract the indexes from the pair
    dept1, dept2 = pair
    
    # split on _ and get index
    '''
    temp=units_df[units_df['Department']==dept1]
    dept1_index = int(temp['id'])
    
    temp=units_df[units_df['Department']==dept2]
    dept2_index = int(temp['id'])
    '''
        
    dept1_index = int(dept1.split("_")[1])
    dept2_index = int(dept2.split("_")[1])
    
    # get the feature matrix of the document
    dept1_fm = feature_matrix.toarray()[dept1_index]
    dept2_fm = feature_matrix.toarray()[dept2_index]
    
    # compute cosine similarity manually
    manual_cosine_similarity = np.dot(dept1_fm, dept2_fm)
    
    return manual_cosine_similarity


pairwise_cosine_similarity = [compute_cosine_similarity(pair) for pair in pairs]

# create a dataframe
df = pd.DataFrame({'pair': pairs, 'similarity': pairwise_cosine_similarity})
display(df.head())
display(df.tail())


from utils import plot_heatmap

# initialize an empty dataframe grid
df_hm = pd.DataFrame({'ind': range(35), 'cols': range(35), 'vals': pd.Series(np.zeros(35))})

# convert to a matrix
df_hm = df_hm.pivot(index='ind', columns='cols').fillna(0)

# make a copy
df_temp = df.copy()

# convert list of tuples into 2 lists
list1 = []
list2 = []
for item1, item2 in df_temp.pair:
    list1.append(item1)
    list2.append(item2)

# add two columns to df_temp
df_temp['dept1'] = list1
df_temp['dept2'] = list2

# drop the pair as it not needed
df_temp.drop('pair', axis=1, inplace=True)

# extract index so that you can construct pairs
df_temp['dept1'] = df_temp['dept1'].apply(lambda x: int(x.split('_')[-1]))
df_temp['dept2'] = df_temp['dept2'].apply(lambda x: int(x.split('_')[-1]))

# create tuples (0, 1, similarity)
df_temp['pairs'] = list(zip(df_temp.dept1, df_temp.dept2, round(df_temp.similarity, 2)))

# display(df_temp.head())

# to get lower diagnol, swap the rows and cols.
for row, col, similarity in df_temp.pairs:
    df_hm.iloc[col, row] = similarity

ax = plot_heatmap(df_hm, ids, ids)

#ax.savefig('heatmap.png', dpi=400)


# display depts which are most similar and least similar
df.loc[[df.similarity.values.argmax(), df.similarity.values.argmin()]]




