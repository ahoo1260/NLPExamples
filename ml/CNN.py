from keras.layers import Dense,Embedding, Dropout, Activation, Conv1D,GlobalMaxPooling1D
from keras.models import Sequential
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


# ----------------------------------------

def CNN(train_X, train_Y, val_X, val_Y):

    model = Sequential()

    model.add(Embedding(MAX_FEATURES,
                        EMBEDDING_DIM,
                        input_length=MAX_LEN))
    model.add(Dropout(DROP))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(NO_FILTERS_0,
                     FILTER_SIZE_0,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(HIDDEN_DIMS))
    model.add(Dropout(DROP))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history=model.fit(train_X, train_Y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(val_X, val_Y))
    model.save(os.path.join(MODELS_PATH,"modelCNN.h5"))

    return history



