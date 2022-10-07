# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:24:19 2020

@author: poseidon
"""

from keras.preprocessing.text        import Tokenizer
from keras.preprocessing.sequence    import pad_sequences


import numpy                         as np
import pandas                        as pd
from   keras.models                  import Sequential
from   keras.layers                  import Dense
from   keras.layers                  import Embedding
from   keras.layers                  import Flatten
from   keras.layers                  import GlobalMaxPool1D
from   keras.layers                  import Conv1D
from   sklearn.model_selection       import train_test_split
#
#
from   BLAS                         import remove_punctuation
from   BLAS                         import stoppingWords
from   BLAS                         import stemming
from   BLAS                         import length
#
from   BLAS                         import create_embedding_matrix
from   BLAS                         import plot_history
from   BLAS                         import evaluateModel


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Sentiment Analysis
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

filepath_dict = {'yelp':   'sentiment_analysis/yelp_labelled.txt',
                 'amazon': 'sentiment_analysis/amazon_cells_labelled.txt',
                 'imdb':   'sentiment_analysis/imdb_labelled.txt'}


df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)




# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# D A T A
#
# CLEANING & PRE-PROCESSING
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# Step 1. Remove punctuation
df['sentence'] = df['sentence'].apply(remove_punctuation)

# Step 2.Extracting the stopwords from nltk library
#df['sentence'] = df['sentence'].apply(stoppingWords)

# Step 3. Stemming operations
#df['sentence'] = df['sentence'].apply(stemming)







# Convert categorical to int
df['label'] = pd.Categorical(df['label'])
df.label = df.label.cat.codes










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
print('Coverage: %.2f%%\n\n\n' % (100 * nonzero_elements/len(tokenizer.word_index)))


#A = []
#MaxLength = 0
#for i in range(len(trainX)):
#    MaxLength = max( MaxLength, len(trainX[i]) )
#    A += [len(trainX[i])]
#print (MaxLength)
#
#plt.bar(range(len(A)), A, label='Feature length')
#plt.show()



# pad sequence
maxlen = 40
trainX = pad_sequences(trainX, padding='post', maxlen=maxlen)
testΧ  = pad_sequences(testΧ,  padding='post', maxlen=maxlen)







model = Sequential()
model.add(Embedding(input_dim    = vocab_size, 
                    output_dim   = embedding_dim, 
                    input_length = maxlen))
#model.add(Flatten())
model.add(Conv1D(filters=128, kernel_size=5,  padding='same', activation='relu'))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
#model.summary()



    
score = model.fit(trainX, trainY,
                  epochs          = 50, 
                  batch_size      = 128, 
                  verbose         = 1, 
                  validation_data=(testΧ, testY))

# Print statistics
plot_history(score)





# Predictions of the classification model
testPredict = model.predict(testΧ)

# Evaluation of the classification model
evaluateModel(testY, testPredict)
