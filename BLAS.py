# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:31:15 2020

@author: poseidon
"""
import numpy                         as np
import matplotlib.pyplot             as plt
plt.style.use('ggplot')
from   sklearn                       import metrics
from   nltk.corpus                   import stopwords
from   nltk.stem.snowball            import SnowballStemmer

# Funtion to remove punctuation
def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)

# Function to remove stopwords
def stoppingWords(text):
    '''a function for removing the stopword'''
    sw = stopwords.words('english')
    
    # displaying the stopwords
    # np.array(sw)
    # print("Number of stopwords: ", len(sw))    
    
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)

# Funtion to carry out stemming operation
def stemming(text):    
# create an object of stemming function
    stemmer = SnowballStemmer("english")    
    '''a function which stems each word in the given text'''
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text) 

# Function to return the length of text
def length(text):    
    '''a function which returns the length of text'''
    return len(text)





# Functions 
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
                
                
    return embedding_matrix

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Testing acc')
    plt.title('Training and Testing accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Testing loss')
    plt.title('Training and Testing loss')
    plt.legend()
    
def evaluateModel(testY, testPredict):
    
    P = np.zeros(testY.shape[0], dtype=int)
    for i in range( testY.shape[0] ):
        if (testPredict[i] < 0.5):
            P[i] = 0
        else:
            P[i] = 1
    CM = metrics.confusion_matrix(testY, P)
    
    Accuracy             = metrics.accuracy_score(testY, P)
    F1                   = metrics.f1_score(testY, P)
    fpr, tpr, thresholds = metrics.roc_curve(testY, P)
    AUC                  = metrics.auc(fpr, tpr)

    # Print results
    print('Accuracy = %.2f%%' % (100*Accuracy))
    print('AUC      = %.5f'   % AUC)
    print('F1       = %.5f'   % F1)
    print(CM)