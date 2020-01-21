#import modules
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from source.ml.CNN import CNN
from source.util.data_helpers import cleanData
from source.util.embedding import getEmbeddingMatrix
import configparser
from source.ml.LSTM import RNN
import tensorflow
from keras.datasets import imdb

#define constants:
from source.visualization.ploting import plotTwoLists

config=configparser.ConfigParser()
config.read('config.txt')
TRAIN_FILE_PATH=config.get('source-config', 'TRAIN_FILE_PATH')
TEST_FILE_PATH=config.get('source-config', 'TEST_FILE_PATH')


MAX_FEATURES=config.getint('source-config', 'MAX_FEATURES')
MAX_LEN = config.getint('source-config', 'MAX_LEN')
EMBEDDING_FILE_PATH=config.get('source-config', 'EMBEDDING_FILE_PATH')

def getBinaryPredictions(preds):
    predictions=[]
    for pred in preds:
        if pred<0.5:
            predictions.append(0)
        else:
            predictions.append(1)
    return predictions

#
# load data
x_train_df= pd.read_csv(TRAIN_FILE_PATH)
test_df=pd.read_csv(TEST_FILE_PATH)

train_df, val_df = train_test_split(x_train_df, test_size=0.2, random_state=2018)

train_X = train_df["review"]
val_X = val_df["review"]

test_X=test_df["review"]

train_Y = train_df['sentiment']
val_Y = val_df['sentiment']
test_Y=test_df['sentiment']



# (train_X, train_Y), (test_X, test_Y) = imdb.load_data(num_words=MAX_FEATURES)

train_X = cleanData(train_X)
val_X = cleanData(val_X)
test_X=cleanData(test_X)

# Tokenize the sentences original----------------------------------------------------------------------------------
tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X=tokenizer.texts_to_sequences(test_X)
vocab_size = len(tokenizer.word_index) + 1

train_X = pad_sequences(train_X, maxlen=MAX_LEN)
val_X = pad_sequences(val_X, maxlen=MAX_LEN)
test_X=pad_sequences(test_X,maxlen=MAX_LEN)

#Embedding----------------------------------------------------------------------------------------------------------
embedding_matrix=getEmbeddingMatrix(EMBEDDING_FILE_PATH, tokenizer, MAX_FEATURES)

#call from ml a method----------------------------------------------------------------------------------------------
cnnModel=CNN( train_X, train_Y, val_X, val_Y)
preds_cnn=cnnModel.model.predict(test_X)
predictions=getBinaryPredictions(preds_cnn)
plotTwoLists(test_Y,predictions,'CNN')


rnnModel=RNN(train_X, train_Y, val_X, val_Y)
preds_rnn=rnnModel.model.predict(test_X)
predictions=getBinaryPredictions(preds_rnn)
plotTwoLists(test_Y,predictions,'RNN')
