# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:09:37 2020

@author: poseidon
"""

import nltk
import urllib.request



# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Tokenize Text Using NLTK
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=



#To tokenize this text to sentences, we will use sentence tokenizer:
from nltk.tokenize import sent_tokenize

mytext = "Hello Adam, how are you? I hope everything is going well. Today is a good day, see you dude."

print(sent_tokenize(mytext))
	

#To tokenize this text to words
from nltk.tokenize import word_tokenize
 
mytext = "Hello Mr. Adam, how are you? I hope everything is going well. Today is a good day, see you dude."
 
print(word_tokenize(mytext))



# Get Synonyms from WordNet
from nltk.corpus import wordnet
 
syn = wordnet.synsets("small")
print('\n\n\n\n') 
print ('Word: small')
print('Definition: ', syn[0].definition())
#print('Synonyms:   ', syn[0].examples())
synonyms = []
for syn in wordnet.synsets('small'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print('Synonyms: ', synonyms)

antonyms = []
for syn in wordnet.synsets("small"):
    for l in syn.lemmas():
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print('Antonyms: ', antonyms)










# NLTK Word Stemming
#
# Word stemming means removing affixes from words and return the root word. 
# Ex: The stem of the word working => work.
# 
# Search engines use this technique when indexing pages, 
# so many people write different versions for the same word 
# and all of them are stemmed to the root word.
#
# There are many algorithms for stemming, but the most 
# used algorithm is Porter stemming algorithm.
from nltk.stem import PorterStemmer
 
stemmer = PorterStemmer() 
print(stemmer.stem('working'))









# Lemmatizing Words Using WordNet
# 
# Word lemmatizing is similar to stemming, but the difference 
# is the result of lemmatizing is a real word.
#
# Unlike stemming, when you try to stem some words, it will 
# result in something like this:
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
print(stemmer.stem('increases'))



# Stemming and Lemmatization Difference
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()
 
print(stemmer.stem('stones'))
print(stemmer.stem('speaking'))
print(stemmer.stem('bedroom'))
print(stemmer.stem('jokes'))
print(stemmer.stem('lisa'))
print(stemmer.stem('purple'))
 
print('----------------------')
 
print(lemmatizer.lemmatize('stones'))
print(lemmatizer.lemmatize('speaking'))
print(lemmatizer.lemmatize('bedroom'))
print(lemmatizer.lemmatize('jokes'))
print(lemmatizer.lemmatize('lisa'))
print(lemmatizer.lemmatize('purple'))



# Stemming works on words without knowing its context and 
# that’s why stemming has lower accuracy and faster than lemmatization.
#
# In my opinion, lemmatizing is better than stemming. 
# Word lemmatizing returns a real word even if it’s not the 
# same word, it could be a synonym, but at least it’s a real word.
#
# Sometimes you don’t care about this level of accuracy 
# and all you need is speed, in this case, stemming is better.