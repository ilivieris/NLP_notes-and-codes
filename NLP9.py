# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:31:39 2020

@author: poseidon
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import matplotlib
from matplotlib import pyplot as plt


from BLAS import remove_punctuation
from BLAS import stoppingWords
from BLAS import stemming
from BLAS import length
# import seaborn as sns
#matplotlib inline
#config InlineBackend.figure_format = 'retina'





# Step 1. Loading and inspecting data
column_names = ['label', 'sentence']
data = pd.read_csv('data/spam.csv', names=column_names, skiprows=[0], sep='\t')

# Count spams and ham emails
nHAM  = data[data['label'] == 'ham'].shape[0]
nSPAM = data[data['label'] == 'spam'].shape[0]


# bar plot of the 2 classes
plt.bar(10, nHAM,  2, label="ham")
plt.bar(15, nSPAM, 3, label="spam")
plt.legend()
plt.ylabel('Number of examples')
plt.title('Propoertion of examples')
plt.show()

# Remove punctuation
data['sentence'] = data['sentence'].apply(remove_punctuation)

# Extracting the stopwords from nltk library
sw = stopwords.words('english')
# displaying the stopwords
np.array(sw)
print("Number of stopwords: ", len(sw))

# Remove stop words
data['sentence'] = data['sentence'].apply(stoppingWords)








# Collect vocabulary count
# Use word counts as feature for NLP since tf-idf is a better metric

# create a count vectorizer object
count_vectorizer = CountVectorizer()
# fit the count vectorizer using the text data
count_vectorizer.fit(data['sentence'])
# collect the vocabulary items used in the vectorizer
dictionary = count_vectorizer.vocabulary_.items()  





# Store the vocab and counts in a pandas dataframe

# lists to store the vocab and counts
vocab = []
count = []
# iterate through each vocab and count append the value to designated lists
for key, value in dictionary:
    vocab.append(key)
    count.append(value)
# store the count in panadas dataframe with vocab as index
vocab_bef_stem = pd.Series(count, index=vocab)
# sort the dataframe
vocab_bef_stem = vocab_bef_stem.sort_values(ascending=False)







# Step 2. Stemming operations

# Stemming operation bundles together words of same root. 
# E.g. stem operation bundles "response" and "respond" into a common "respon"

# create an object of stemming function
stemmer = SnowballStemmer("english")

data['sentence'] = data['sentence'].apply(stemming)
data.head(10)



# Top words after stemming operation

# Collect vocabulary count
# create the object of tfid vectorizer
tfid_vectorizer = TfidfVectorizer("english")
# fit the vectorizer using the text data
tfid_vectorizer.fit(data['sentence'])
# collect the vocabulary items used in the vectorizer
dictionary = tfid_vectorizer.vocabulary_.items()  






# Apply the function to each example
data['length'] = data['sentence'].apply(length)
data.head(10)





# Extracting data of each class
HAM_data  = data[data['label'] == 'ham']
SPAM_data = data[data['label'] == 'spam']


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
bins = 500
plt.hist(HAM_data['length'], alpha = 0.6, bins=bins, label='ham')
plt.hist(SPAM_data['length'], alpha = 0.8, bins=bins, label='spam')
plt.xlabel('length')
plt.ylabel('numbers')
plt.legend(loc='upper right')
plt.grid()
plt.show()


##  Top words of each email and their count
## create the object of tfid vectorizer
#HAM_tfid_vectorizer = TfidfVectorizer("english")
## fit the vectorizer using the text data
#HAM_tfid_vectorizer.fit(HAM_data['sentence'])
## collect the vocabulary items used in the vectorizer
#HAM_dictionary = HAM_tfid_vectorizer.vocabulary_.items()
#
## lists to store the vocab and counts
#vocab = []
#count = []
## iterate through each vocab and count append the value to designated lists
#for key, value in HAM_dictionary:
#    vocab.append(key)
#    count.append(value)
## store the count in panadas dataframe with vocab as index
#HAM_vocab = pd.Series(count, index=vocab)
## sort the dataframe
#HAM_vocab = HAM_vocab.sort_values(ascending=False)
## plot of the top vocab
#top_vacab = HAM_vocab.head(20)
#top_vacab.plot(kind = 'barh', figsize=(5,10))

