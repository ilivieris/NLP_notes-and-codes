# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:09:37 2020

@author: poseidon
"""

import nltk
import urllib.request



# Learn how to identify what the web page is about using NLTK in Python
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# Import a web page
# urllib module will help us to crawl the webpage
response =  urllib.request.urlopen('https://www.math.upatras.gr/en/')
html     =  response.read()


# Beautiful Soup which is a Python library for pulling data out 
# of HTML and XML files. We will use beautiful soup to clean our 
# webpage text of HTML tags.
from bs4 import BeautifulSoup
soup = BeautifulSoup(html,'html5lib')
text = soup.get_text(strip = True)


# Clean text from the crawled web page, 
# let’s convert the text into tokens.
tokens = [t for t in text.split()]




# Count word Frequency
# =-=-=-=-=-=-=-=-=-=-=
# nltk offers a function FreqDist() which will do the job for us. 
# Also, we will remove stop words (a, at, the, for etc) from our web 
# page as we don't need them to hamper our word frequency count. 
# We will plot the graph for most frequently occurring words in 
# the webpage in order to get the clear picture of the context of the web page
from nltk.corpus import stopwords
sr = stopwords.words('english')

clean_tokens = tokens[:]
for token in tokens:
    if token in stopwords.words('english'):
        clean_tokens.remove(token)

freq = nltk.FreqDist(clean_tokens)
for key,val in freq.items():
    print(str(key) + ':' + str(val))

freq.plot(10, cumulative=False)



