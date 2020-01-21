
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import os

import configparser
config=configparser.ConfigParser()
config.read('config.txt')
MAX_FEATURES = config.getint('source-config', 'MAX_FEATURES')
MAX_LEN = config.getint('source-config', 'MAX_LEN')

FILTER_SIZE_0 = config.getint('source-config', 'FILTER_SIZE_0')

NO_FILTERS_0 = config.getint('source-config', 'NO_FILTERS_0')
EMBEDDING_DIM = config.getint('source-config', 'EMBEDDING_DIM')
HIDDEN_DIMS = config.getint('source-config', 'HIDDEN_DIMS')
DROP = config.getfloat('source-config', 'DROP')
BATCH_SIZE = config.getint('source-config', 'BATCH_SIZE')
EPOCHS = config.getint('source-config', 'EPOCHS')
MODELS_PATH = config.get('source-config', 'MODELS_PATH')

def RNN(train_X, train_Y, val_X, val_Y):
    inputs = Input(name='inputs',shape=[MAX_LEN])
    layer = Embedding(MAX_FEATURES,EMBEDDING_DIM,input_length=MAX_LEN)(inputs)
    layer=Dropout(DROP)(layer)
    layer = LSTM(100)(layer)
    # layer = Dense(256,name='FC1')(layer)
    layer = Dropout(DROP)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_X, train_Y, batch_size=BATCH_SIZE, epochs=EPOCHS)
    model.save(os.path.join(MODELS_PATH,"modelLSTM.h5"))
    return history