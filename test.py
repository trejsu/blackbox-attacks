import os

import keras.backend as K
import numpy as np
from tqdm import tqdm

from mnist import data_mnist
from mnist import load_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    x = K.placeholder((None, 28, 28, 1))

    model = load_model(args.model)
    logits = model(x)
    prediction = K.softmax(logits)

    if args.mnist:
        _, _, X, Y = data_mnist(one_hot=False)
        accuracy = get_accuracy(X, Y, prediction, x)
        print(f'accuracy = {accuracy}')

    elif args.n is None:
        with np.load(args.dataset) as data:
            X = data['drawings']
            Y = data['Y'].reshape(-1, )

        accuracy = get_accuracy(X, Y, prediction, x)
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


def get_accuracy(X, Y, prediction, x):
    predictions = \
        K.get_session().run([prediction], feed_dict={x: X, K.learning_phase(): 0})[0]
    argmax = np.argmax(predictions, axis=1)
    equal = argmax == Y
    accuracy = np.mean(equal)
    return accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="target model",
                        default="models_newest_keras/modelA")
    parser.add_argument("--dataset", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--targeted", type=bool, default=False)
    parser.add_argument("--attack", type=bool, default=False)
    parser.add_argument("--mnist", type=bool, default=False)
    parser.add_argument("--n", type=int, default=None)

    args = parser.parse_args()

    main()
