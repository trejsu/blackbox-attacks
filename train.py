import keras
from keras import backend as K
from keras.models import save_model

from mnist import *
from tf_utils import tf_train, tf_test_error_rate

FLAGS = tf.flags.FLAGS


def main(args):
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"
    set_mnist_flags()

    with tf.device('/gpu:0'):
        tf.flags.DEFINE_integer('NUM_EPOCHS', args.epochs, 'Number of epochs')
        tf.flags.DEFINE_integer('MODEL_TYPE', args.type, 'Type of the model')

        # Get MNIST test data
        X_train, Y_train, X_test, Y_test = data_mnist()

        data_gen = data_gen_mnist(X_train)

        x = K.placeholder((None,
                           FLAGS.IMAGE_ROWS,
                           FLAGS.IMAGE_COLS,
                           FLAGS.NUM_CHANNELS
                           ))

        y = K.placeholder(shape=(None, FLAGS.NUM_CLASSES))

        model = model_mnist(type=args.type)

        # print(model.summary())

        # Train an MNIST model
        tf_train(x, y, model, X_train, Y_train, data_gen, None, None)

        # Finally print the result!
        _, _, test_error = tf_test_error_rate(model, x, X_test, Y_test)
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
    args = parser.parse_args()

    args.type = 0
    args.epochs = 6

    main(args)
