# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:19:44 2020

@author: poseidon
"""

from keras.preprocessing.text        import Tokenizer
from keras.preprocessing.sequence    import pad_sequences


import numpy                         as np
import pandas                        as pd
import matplotlib.pyplot             as plt
plt.style.use('ggplot')
from   keras.models                  import Sequential
from   keras.layers                  import Dense
from   keras.layers                  import Embedding
from   keras.layers                  import Flatten
from   keras.layers                  import GlobalMaxPool1D
from   keras.layers                  import Conv1D
from sklearn.model_selection         import train_test_split
from sklearn                         import metrics


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

    plt.figure(figsize=(12, 5))
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
    print('F1       = %.5f'   %  F1)
    print(CM)



# Sentiment Labelled Sentences Data Set (UCI)
# This data set includes labeled reviews from 
# 1. IMDb, 
# 2. Amazon
# 3. Yelp. 
# Each review is marked with a score of 0 for a negative sentiment 
# or 1 for a positive sentiment.
filepath_dict = {'yelp':   'sentiment_analysis/yelp_labelled.txt',
                 'amazon': 'sentiment_analysis/amazon_cells_labelled.txt',
                 'imdb':   'sentiment_analysis/imdb_labelled.txt'}


df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)






# Get Sentences and Ratings
Sentences = df['sentence'].values
Ratings   = df['label'].values


Sentences_train, Sentences_test, trainY, testY = train_test_split(Sentences, Ratings, test_size=0.1, random_state=1000)



# Use keras-tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(Sentences_train)

trainX = tokenizer.texts_to_sequences(Sentences_train)
testΧ  = tokenizer.texts_to_sequences(Sentences_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index







# Retrieve the embedding matrix
embedding_dim = 50
embedding_matrix = create_embedding_matrix('data/glove.6B.50d.txt', tokenizer.word_index, embedding_dim)


nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print('Coverage: %.2f%%' % (100 * nonzero_elements/len(tokenizer.word_index)))








# pad sequence
maxlen = 100
trainX = pad_sequences(trainX, padding='post', maxlen=maxlen)
testΧ   = pad_sequences(testΧ, padding='post', maxlen=maxlen)









embedding_dim = 50

model = Sequential()
model.add(Embedding(input_dim    = vocab_size, 
                    output_dim   = embedding_dim, 
                    input_length = maxlen))
#model.add(Flatten())
model.add(Conv1D(filters=128, kernel_size=5,  padding='same', activation='relu'))
model.add(GlobalMaxPool1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.summary()



    
score = model.fit(trainX, trainY,
                  epochs          = 50, 
                  batch_size      = 128, 
                  verbose         = 0, 
                  validation_data=(testΧ, testY))








# Print statistics
plot_history(score)

# Predictions of the classification model
testPredict = model.predict(testΧ)

# Evaluation of the classification model
evaluateModel(testY, testPredict)
