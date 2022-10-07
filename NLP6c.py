# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:49:54 2020

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


class SpamClassifier(object):
    def __init__(self):
        self.maxlen = 40
        self.embedding_dim = 50
    
    def load_data(self):    
        filepath_dict = {'yelp':   'sentiment_analysis/yelp_labelled.txt',
                         'amazon': 'sentiment_analysis/amazon_cells_labelled.txt',
                         'imdb':   'sentiment_analysis/imdb_labelled.txt'}


        df_list = []
        for source, filepath in filepath_dict.items():
            df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
            df['source'] = source  # Add another column filled with the source name
            df_list.append(df)

        df = pd.concat(df_list)

        # Convert categorical to int
        df['label'] = pd.Categorical(df['label'])
        df.label = df.label.cat.codes

        self.data = df
        
    def split_data(self):
        # Get Sentences and Ratings
        Sentences = self.data['sentence'].values
        Ratings   = self.data['label'].values
        
        self.Sentences_train, self.Sentences_test, self.trainY, self.testY = train_test_split(Sentences, Ratings, test_size=0.1, random_state=1000)

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

    def tokenize(self):
        # Use keras-tokenizer
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(self.Sentences_train)

        trainX = tokenizer.texts_to_sequences(self.Sentences_train)
        testΧ  = tokenizer.texts_to_sequences(self.Sentences_test)

        self.vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index


        # Retrieve the embedding matrix
        embedding_matrix = SpamClassifier.create_embedding_matrix('data/glove.6B.50d.txt', tokenizer.word_index, self.embedding_dim)


        nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
        print('Coverage: %.2f%%\n\n\n' % (100 * nonzero_elements/len(tokenizer.word_index)))

        self.trainX = pad_sequences(trainX, padding='post', maxlen=self.maxlen)
        self.testΧ  = pad_sequences(testΧ,  padding='post', maxlen=self.maxlen)


    def define_model(self):    
        model = Sequential()
        model.add(Embedding(input_dim   = self.vocab_size, 
                           output_dim   = self.embedding_dim, 
                           input_length = self.maxlen))
        #model.add(Flatten())
        model.add(Conv1D(filters=128, kernel_size=5,  padding='same', activation='relu'))
        model.add(GlobalMaxPool1D())
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
        
        
        self.model = model
        
    def train_model(self):
        self.score = self.model.fit(self.trainX, self.trainY,
                     epochs          = 3, 
                     batch_size      = 128, 
                     verbose         = 1, 
                     validation_data=(self.testΧ, self.testY))        
        
    def make_prediction(self):
        # Predictions of the classification model
        self.testPredict = self.model.predict(self.testΧ)

        P = np.zeros(self.testY.shape[0], dtype=int)
        for i in range( self.testY.shape[0] ):
            if (self.testPredict[i] < 0.5):
                P[i] = 0
            else:
                P[i] = 1
        
        self.testPredict = P
        
    def plot_history(self):
        history = self.score
        
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
        plt.show()

    def evaluateModel(self):
    
        CM = metrics.confusion_matrix(self.testY, self.testPredict)
    
        Accuracy             = metrics.accuracy_score(self.testY, self.testPredict)
        F1                   = metrics.f1_score(self.testY, self.testPredict)
        fpr, tpr, thresholds = metrics.roc_curve(self.testY, self.testPredict)
        AUC                  = metrics.auc(fpr, tpr)

        # Print results
        print('\n\n\n')
        print('Accuracy = %.2f%%' % (100*Accuracy))
        print('AUC      = %.5f'   % AUC)
        print('F1       = %.5f'   % F1)
        print(CM)
        
    def main(self):        
        self.load_data()
        self.split_data()
        self.tokenize()
        self.define_model()
        self.train_model()
        self.make_prediction()
        self.plot_history()        
        self.evaluateModel()
        
        
tt = SpamClassifier()        
tt.main()