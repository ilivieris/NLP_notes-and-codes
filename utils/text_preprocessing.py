from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd


def remove_punctuation(text:str = None):
    '''
        Function for removing punctuation

        Parameters
        ----------
        text: input text

        Returns
        -------
        text with removed punctuation
    '''    
    # replacing the punctuations with no space, which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    
    # return the text stripped of punctuation marks
    return text.translate(translator)

def remove_stopping_words(text:str = None):
    '''
        Function for removing the stopword

        Parameters
        ----------
        text: input text

        Returns
        -------
        text with removed stopping words
    '''
    sw = stopwords.words('english')
    
    # removing the stop words and lowercasing the selected words
    text = [word for word in text.split() if word not in sw]
    # joining the list of words with space separator
    return " ".join(text)



def lemmatization(text:str = None):
    '''
        Function for lemmatizing

        Parameters
        ----------
        text: input text

        Returns
        -------
        text after the application of lemmatization
    '''
    lemmatizer = WordNetLemmatizer()

    text = [lemmatizer.lemmatize(word) for word in text.split()]

    return " ".join(text)



def preprocess(Series:pd.Series = None, verbose:bool = True):
    '''
        Data preprocessing
        - Step 1. Apply lower-case
        - Step 2. Remove punctuation
        - Step 3. Extracting the stopwords
        - Step 4. Lemmatization operation

        Parameters
        ----------
        Series: Series containing documents
        verbose: boolean variable for presenting results

        Returns
        -------
        Dataframe with preprocessed documents
    '''
    if verbose: print('Text preprocessing')
    
    # Step 1. Apply lower-case
    Series = Series.apply(lambda x: x.lower())
    if verbose: print('Step 1. Apply lower-case')

    # Step 2. Remove punctuation
    Series = Series.apply(remove_punctuation)
    if verbose: print('Step 2. Remove punctuation')

    # Step 3. Extracting the stopwords
    Series = Series.apply(remove_stopping_words)
    if verbose: print('Step 3. Extracting the stopwords')

    # Step 4. Lemmatization operation
    Series = Series.apply(lemmatization)
    if verbose: print('Step 4. Lemmatization operation')

    return Series