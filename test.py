import os

import keras.backend as K
import numpy as np
from tensorflow.python.platform import flags

from mnist import data_mnist
from mnist import load_model
from mnist import set_mnist_flags
from tqdm import tqdm

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
        result = []

        for i in tqdm(range(1, args.n + 1)):
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
            result.append(accuracy)

        print(result)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="target model", default="models_newest_keras/modelA")
    parser.add_argument("--dataset", type=str, help="Path to the dataset",
                        default='/Users/mchrusci/uj/shaper_data/adversarial/fixed/fgs/fgs-redrawned-%d.npz')
    parser.add_argument("--targeted", type=bool, default=False)
    parser.add_argument("--attack", type=bool, default=True)
    parser.add_argument("--mnist", type=bool, default=False)
    parser.add_argument("--n", type=int, default=100)

    args = parser.parse_args()

    set_mnist_flags()

    main()
