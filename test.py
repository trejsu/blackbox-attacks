import os

import keras.backend as K
import numpy as np
from tensorflow.python.platform import flags

from mnist import data_mnist
from mnist import load_model
from mnist import set_mnist_flags

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

FLAGS = flags.FLAGS


def main():
    x = K.placeholder((None, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))

    model = load_model(args.model)
    logits = model(x)
    prediction = K.softmax(logits)

    if args.mnist:
        _, _, X, Y = data_mnist(one_hot=False)

        predictions = \
        K.get_session().run([prediction], feed_dict={x: X[0:1000], K.learning_phase(): 0})[0]
        argmax = np.argmax(predictions, axis=1)
        equal = argmax == Y[0:1000]
        accuracy = np.mean(equal)
        print(f'accuracy = {accuracy}')
    else:
        N = 20
        result = []

        for i in range(1, N + 1):
            with np.load(args.dataset % i) as data:
                X = data['drawings']
                Y = data['Y'].reshape(-1, )

            predictions = \
            K.get_session().run([prediction], feed_dict={x: X, K.learning_phase(): 0})[0]
            argmax = np.argmax(predictions, axis=1)

            if args.attack and not args.targeted:
                equal = argmax != Y
            else:
                equal = argmax == Y

            accuracy = np.mean(equal)
            print(f'accuracy = {accuracy}')
            result.append(accuracy)

        print(result)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="target model", default="models_newest_keras/modelB")
    parser.add_argument("--dataset", type=str, help="Path to the dataset",
                        default='/Users/mchrusci/uj/shaper_data/activation-distance/single-reward/l2-%d.npz')
    parser.add_argument("--targeted", type=bool, default=False)
    parser.add_argument("--attack", type=bool, default=False)
    parser.add_argument("--mnist", type=bool, default=False)

    args = parser.parse_args()

    set_mnist_flags()

    main()
