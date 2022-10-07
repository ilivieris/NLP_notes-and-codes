# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:23:52 2020

@author: poseidon
"""
import numpy                         as np
import pandas                        as pd
import matplotlib.pyplot             as plt
plt.style.use('ggplot')
import tensorflow
from   tensorflow.keras.models                  import Sequential
from   tensorflow.keras.layers                  import Dense
from   tensorflow.keras.layers                  import LSTM


from sklearn.model_selection         import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn                         import metrics


# Functions 
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
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








# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                T E S T        C A S E
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# Get Sentences and Ratings
Sentences = df['sentence'].values
Ratings   = df['label'].values


Sentences_train, Sentences_test, trainY, testY = train_test_split(Sentences, Ratings, test_size=0.1, random_state=1000)


# Vectorization process
vectorizer = CountVectorizer()
vectorizer.fit(Sentences_train)

trainX = vectorizer.transform(Sentences_train)
testX  = vectorizer.transform(Sentences_test)







#
#X_train = X_train.toarray()
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#
#
#
#X_test = X_test.toarray()
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#
#
#model = Sequential()
#model.add(LSTM(10,  activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
##model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
#model.add(Dense(1,  activation='linear'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
#model.summary()
#
#



model = Sequential()
model.add(Dense(40, input_dim=trainX.shape[1], activation='relu'))
model.add(Dense(1,  activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
model.summary()






    
score = model.fit(trainX, trainY,
                  epochs          = 20, 
                  batch_size      = 128, 
                  verbose         = 1, 
                  validation_data=(testX, testY))


loss, accuracy = model.evaluate(trainX, trainY, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(testX, testY, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))





# Print statistics
plot_history(score)

# Predictions of the classification model
testPredict = model.predict(testX)

# Evaluation of the classification model
evaluateModel(testY, testPredict)



    
    





#for source in df['source'].unique():
#    df_source = df[df['source'] == source]
#    sentences = df_source['sentence'].values
#    y = df_source['label'].values
#
#    sentences_train, sentences_test, y_train, y_test = train_test_split(
#        sentences, y, test_size=0.25, random_state=1000)
#
#    vectorizer = CountVectorizer()
#    vectorizer.fit(sentences_train)
#    X_train = vectorizer.transform(sentences_train)
#    X_test  = vectorizer.transform(sentences_test)
#
##    classifier = LogisticRegression()
##    classifier.fit(X_train, y_train)
#    # Create and fit the LSTM network
#    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#    print (X_train.shape)
#    model = Sequential()
#    model.add(LSTM(30,  activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
#    #model.add(Dense(8, activation='relu'))
#    model.add(Dense(1,  activation='linear'))
#    model.compile(loss='mean_squared_error', optimizer='adam')
#
#
#
#    
#    score = model.fit(X_train, y_train, 
#                      epochs          = 100, 
#                      batch_size      = 128, 
#                      verbose         = 0, 
#                      validation_data = (X_test, y_test))
#    
#    score = classifier.score(X_test, y_test)
#    print('Accuracy for {} data: {:.4f}'.format(source, score))
#    break























## Imagine you have the following two sentences:
#sentences = ['John likes ice cream', 'John hates chocolate']
#for i, x in enumerate(sentences):
#    print ('Sentence: %i -> %s' % (i,x))
#print('\n\n\n')
#
#from sklearn.feature_extraction.text import CountVectorizer
#
#vectorizer = CountVectorizer(min_df=0, lowercase=False)
#vectorizer.fit(sentences)
#for word in sorted(vectorizer.vocabulary_):
#    print('%s. -> %s' % (vectorizer.vocabulary_[word], word))
#
#vectorizer.transform(sentences).toarray()