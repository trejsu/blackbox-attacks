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
        x_train, y_train, x_test, y_test = data_mnist() if args.dataset == "mnist" \
            else data(
            path=args.dataset,
            representation=args.representation,
            test_path=args.test_path,
            n=args.n
        )

        data_gen = data_gen_mnist(x_train)

        x = K.placeholder(args.x_dim)
        y = K.placeholder(shape=(None, 10))

        model = model_mnist(type=args.type)

        tf_train(x, y, model, x_train, y_train, data_gen, None, None)

        _, _, test_error = tf_test_error_rate(model(x), x, x_test, y_test)
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
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--x-dim", nargs='+', type=int)
    parser.add_argument("--representation", action="store_true")
    parser.add_argument("--test-path", type=str)
    parser.add_argument("--n", type=int)
    args = parser.parse_args()

    args.x_dim = tuple(args.x_dim)

    args.type = 0
    args.epochs = 6

    assert not args.representation or args.n is not None

    main(args)
