from __future__ import print_function
from autokeras.image_supervised import load_image_dataset
from autokeras import ImageClassifier
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # print(x_train.shape, x_train[0].shape, x_train.reshape(x_train.shape + (1,)).shape)

    NB_EPOCH = 200
    BATCH_SIZE = 128
    VERBOSE = 1
    NB_CLASSES = 3
    OPTIMIZER = RMSprop() # optimizer, explainedin this chapter
    N_HIDDEN = 128
    VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
    DROPOUT = 0.3

    RESHAPED = 7575

    x_train, y_train = load_image_dataset(csv_file_path="train.csv",
                                          images_path="images")
    # print('x_train', x_train[0].shape, type(x_train))

    # x_train = x_train.reshape(x_train.shape[0], 396, 532, 4)

    print('x_train', x_train.shape)
    # print('y_train', y_train.shape)

    x_test, y_test = load_image_dataset(csv_file_path="test.csv",
                                        images_path="images")
    # print(x_test.shape)
    # print(y_test.shape)

    x_train = x_train.reshape(3077, RESHAPED)
    x_test = x_test.reshape(1025, RESHAPED)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = pd.Series(y_train)
    y_train = pd.to_numeric(y_train).values.astype(int)
    y_test = pd.Series(y_test)
    y_test = pd.to_numeric(y_test).values.astype(int)

    print(x_train.shape, y_train.shape, y_train[0])

    y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    print(x_train.shape, y_train.shape, y_train[0])

    model = Sequential()
    model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(N_HIDDEN))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=OPTIMIZER,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                        verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

    score = model.evaluate(x_test, y_test, verbose=VERBOSE)
    print("\nTest score:", score[0])
    print('Test accuracy:', score[1])

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # clf = ImageClassifier(verbose=True, augment=False)
    # clf.fit(x_train, y_train)
    # clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    # y = clf.evaluate(x_test, y_test)
    # print(y)