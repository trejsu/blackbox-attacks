import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import keras
from keras import backend as K
from keras.models import save_model

from mnist import *
from tf_utils import tf_train, tf_test_error_rate
import tensorflow as tf


def main(args):
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"

    with tf.device('/gpu:0'):
        # Get MNIST test data
        X_train, Y_train, X_test, Y_test = data_mnist() if args.dataset == "mnist" \
            else data(path=args.dataset)

        # N = 1000
        # C = N // 10
        #
        # X_train_reduced = np.empty((N, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        # Y_train_reduced = np.empty((N, Y_train.shape[1]))
        #
        # for i in range(10):
        #     indexes = np.where(np.argmax(Y_train, axis=1) == i)
        #     X_train_reduced[i * C:i * C + C] = X_train[indexes][0:C]
        #     Y_train_reduced[i * C:i * C + C] = Y_train[indexes][0:C]
        #
        # np.random.seed(666)
        # np.random.shuffle(X_train_reduced)
        # np.random.seed(666)
        # np.random.shuffle(Y_train_reduced)
        #
        # X_train = X_train_reduced
        # Y_train = Y_train_reduced
        #
        # argmax = np.argmax(Y_train, axis=1)
        #
        # for i in range(10):
        #     assert np.sum(argmax == i) == C, f'{i} = {np.sum(argmax == i)}'
        #
        # np.savez(f'mnist-{N}', X=X_train, Y=Y_train)

        data_gen = data_gen_mnist(X_train)

        x = K.placeholder((None, 28, 28, 1))
        y = K.placeholder(shape=(None, 10))

        model = model_mnist(type=args.type)

        # print(model.summary())

        # Train an MNIST model
        tf_train(x, y, model, X_train, Y_train, data_gen, None, None)

        # Finally print the result!
        _, _, test_error = tf_test_error_rate(model(x), x, X_test, Y_test)
        print('Test error: %.1f%%' % test_error)
        save_model(model, args.model)
        json_string = model.to_json()
        try:
            with open(args.model + '.json', 'w') as f:
                f.write(json_string)
        except Exception:
            print(json_string)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to model")
    # parser.add_argument("--type", type=int, help="model type", default=1)
    # parser.add_argument("--epochs", type=int, default=6, help="number of epochs")
    parser.add_argument("--dataset", default="mnist")
    args = parser.parse_args()

    args.type = 0
    args.epochs = 6

    main(args)
