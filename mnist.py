import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, Convolution2D
from keras.models import Sequential, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def data_mnist(one_hot=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if one_hot:
        y_test, y_train = make_one_hot(y_test, y_train)
    x_train, x_test = preprocess_mnist(x_test, x_train)
    return x_train, y_train, x_test, y_test


def make_one_hot(y_test, y_train):
    y_train = np_utils.to_categorical(y_train, 10).astype(np.float32)
    y_test = np_utils.to_categorical(y_test, 10).astype(np.float32)
    return y_test, y_train


def preprocess_mnist(x_test, x_train):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    return x_train, x_test


def preprocess_representation(x_test, x_train, n):
    x_train = x_train[:, 0:n].reshape(-1, 8)
    x_test = x_test[:, 0:n].reshape(-1, 8)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    n_components = 7
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(x_train_scaled)
    x_train_reduced = pca.transform(x_train_scaled)
    x_test_reduced = pca.transform(x_test_scaled)

    return x_train_reduced.reshape(-1, n, n_components), x_test_reduced.reshape(-1, n, n_components)


def data(path, representation=False, test_path=None, one_hot=True, n=None):
    if representation:
        with np.load(test_path) as dataset:
            x_test = dataset['drawings']
            y_test = dataset['Y']
    else:
        (_, _), (x_test, y_test) = mnist.load_data()

    with np.load(path) as dataset:
        x_train = dataset['drawings']
        y_train = dataset['Y']

    if one_hot:
        y_test, y_train = make_one_hot(y_test, y_train)

    x_train, x_test = preprocess_representation(x_test, x_train, n) if representation \
        else preprocess_mnist(x_test, x_train)

    return x_train, y_train, x_test, y_test


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
