import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, Convolution2D
from keras.models import Sequential, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils


def data_mnist(one_hot=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return preprocess(x_test, x_train, one_hot, y_test, y_train)


def preprocess(x_test, x_train, one_hot, y_test, y_train):
    print(f'y_train.shape = {y_train.shape}')
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('X_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print("Loaded MNIST test data.")
    if one_hot:
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, 10).astype(np.float32)
        y_test = np_utils.to_categorical(y_test, 10).astype(np.float32)
    return x_train, y_train, x_test, y_test


def data(path, one_hot=True):
    (_, _), (x_test, y_test) = mnist.load_data()

    with np.load(path) as dataset:
        x_train = dataset['drawings']
        y_train = dataset['Y']

    return preprocess(x_test, x_train, one_hot, y_test, y_train)


def modelA():
    model = Sequential()
    model.add(Convolution2D(64, (5, 5), padding='valid'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, (5, 5)))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(10))
    return model


def modelB():
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(28, 28, 1)))
    model.add(Convolution2D(64, 8, 8, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 6, 6, subsample=(2, 2), border_mode='valid'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 5, 5, subsample=(1, 1)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(10))
    return model


def modelC():
    model = Sequential()
    model.add(Convolution2D(128, 3, 3,
                            border_mode='valid',
                            input_shape=(28,
                                         28,
                                         1)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(10))
    return model


def modelD():
    model = Sequential()

    model.add(Flatten(input_shape=(28,
                                   28,
                                   1)))

    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    return model


def modelE():
    model = Sequential()

    model.add(Flatten(input_shape=(28,
                                   28,
                                   1)))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))

    model.add(Dense(10))

    return model


def modelF():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3,
                            border_mode='valid',
                            input_shape=(28,
                                         28,
                                         1)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))

    model.add(Dense(10))

    return model


def model_mnist(type=1):
    """
    Defines MNIST model using Keras sequential model
    """
    models = [modelA, modelB, modelC, modelD, modelE, modelF]

    return models[type]()


def data_gen_mnist(X_train):
    datagen = ImageDataGenerator()

    datagen.fit(X_train)
    return datagen


def load_model(model_path, type=0):
    try:
        with open(model_path + '.json', 'r') as f:
            json_string = f.read()
            model = model_from_json(json_string)
    except IOError:
        model = model_mnist(type=type)

    model.load_weights(model_path)
    return model
