from os.path import basename

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.utils import np_utils

from attack_utils import gen_grad_ens
from fgs import symbolic_fgs, symbolic_fg
from mnist import data_mnist, set_mnist_flags, load_model
from tf_utils import tf_test_error_rate, batch_eval

FLAGS = tf.flags.FLAGS


def gen_grad_cw(x, logits, y):
    real = tf.reduce_sum(y * logits, 1)
    other = tf.reduce_max((1 - y) * logits - (y * 10000), 1)
    loss = tf.maximum(0.0, real - other + args.kappa)
    grad = -1.0 * K.gradients(loss, [x])[0]
    return grad


def main(attack, src_model_name, target_model_name):
    np.random.seed(0)
    tf.set_random_seed(0)

    tf.flags.DEFINE_integer('BATCH_SIZE', 10, 'Size of batches')
    set_mnist_flags()

    dim = FLAGS.IMAGE_ROWS * FLAGS.IMAGE_COLS * FLAGS.NUM_CHANNELS

    x = K.placeholder((None,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS,
                       FLAGS.NUM_CHANNELS))

    y = K.placeholder((None, FLAGS.NUM_CLASSES))

    _, _, X_test, Y_test = data_mnist()
    Y_test_uncat = np.argmax(Y_test, axis=1)

    # source model for crafting adversarial examples
    src_model = load_model(src_model_name)

    # model(s) to target
    target_model = load_model(target_model_name)

    # simply compute test error
    if attack == "test":
        _, _, err = tf_test_error_rate(src_model, x, X_test, Y_test)
        print('{}: {:.1f}'.format(basename(src_model_name), err))
        _, _, err = tf_test_error_rate(target_model, x, X_test, Y_test)
        print('{}: {:.1f}'.format(basename(target_model_name), err))

        return

    if args.targeted_flag == 1:
        targets = []
        allowed_targets = list(range(FLAGS.NUM_CLASSES))
        for i in range(len(Y_test)):
            allowed_targets.remove(Y_test_uncat[i])
            targets.append(np.random.choice(allowed_targets))
            allowed_targets = list(range(FLAGS.NUM_CLASSES))
        targets = np.array(targets)
        print(targets)
        targets_cat = np_utils.to_categorical(targets, FLAGS.NUM_CLASSES).astype(np.float32)
        Y_test = targets_cat

    logits = src_model(x)
    print('logits', logits)

    if args.loss_type == 'xent':
        loss, grad = gen_grad_ens(x, logits, y)
        assert grad is not None
    elif args.loss_type == 'cw':
        grad = gen_grad_cw(x, logits, y)
    if args.targeted_flag == 1:
        grad = -1.0 * grad

    for eps in eps_list:
        # FGSM and RAND+FGSM one-shot attack
        if attack in ["fgs", "rand_fgs"] and args.norm == 'linf':
            assert grad is not None
            adv_x = symbolic_fgs(x, grad, eps=eps)
        elif attack in ["fgs", "rand_fgs"] and args.norm == 'l2':
            adv_x = symbolic_fg(x, grad, eps=eps)

        # iterative FGSM
        if attack == "ifgs":
            l = 1000
            X_test = X_test[0:l]
            Y_test = Y_test[0:l]

            adv_x = x
            # iteratively apply the FGSM with small step size
            for i in range(args.num_iter):
                adv_logits = src_model(adv_x)

                if args.loss_type == 'xent':
                    loss, grad = gen_grad_ens(adv_x, adv_logits, y)
                elif args.loss_type == 'cw':
                    grad = gen_grad_cw(adv_x, adv_logits, y)
                if args.targeted_flag == 1:
                    grad = -1.0 * grad

                adv_x = symbolic_fgs(adv_x, grad, args.delta, True)
                r = adv_x - x
                r = K.clip(r, -eps, eps)
                adv_x = x + r

            adv_x = K.clip(adv_x, 0, 1)

        print('Generating adversarial samples')
        X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]

        avg_l2_perturb = np.mean(np.linalg.norm((X_adv - X_test).reshape(len(X_test), dim), axis=1))

        # white-box attack
        l = len(X_adv)
        print('Carrying out white-box attack')
        preds_adv, orig, err = tf_test_error_rate(src_model, x, X_adv, Y_test[0:l])
        if args.targeted_flag == 1:
            err = 100.0 - err
        print('{}->{}: {:.1f}'.format(src_model_name, src_model_name, err))

        # black-box attack
        if target_model_name is not None:
            print('Carrying out black-box attack')
            preds, _, err = tf_test_error_rate(target_model, x, X_adv, Y_test)
            if args.targeted_flag == 1:
                err = 100.0 - err
            print('{}->{}: {:.1f}, {}, {} {}'.format(src_model_name,
                                                     basename(target_model_name), err,
                                                     avg_l2_perturb, eps, attack))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("attack", help="name of attack",
                        choices=["test", "fgs", "ifgs", "rand_fgs", "CW_ens"])
    parser.add_argument('src_model', type=str,
                        help='source model for creating examples')
    parser.add_argument('target_model', help="target model for attack")
    parser.add_argument("--eps", type=float, default=None,
                        help="FGS attack scale")
    parser.add_argument("--loss_type", type=str, default='cw',
                        help="Type of loss to use")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="RAND+FGSM random perturbation scale")
    parser.add_argument("--delta", type=float, default=0.01,
                        help="Iterated FGS step size")
    parser.add_argument("--num_iter", type=int, default=40,
                        help="Iterated FGS step size")
    parser.add_argument("--kappa", type=float, default=100.0,
                        help="CW attack confidence")
    parser.add_argument("--norm", type=str, default='linf',
                        help="Norm to use for attack")
    parser.add_argument("--targeted_flag", type=int, default=0,
                        help="Carry out targeted attack")

    args = parser.parse_args()

    if args.eps is None:
        if args.norm == 'linf':
            eps_list = [0.3]
            if args.attack == "ifgs":
                eps_list = [0.3]
        elif args.norm == 'l2':
            eps_list = list(np.linspace(0.0, 2.0, 5))
            eps_list.extend(np.linspace(2.5, 9.0, 14))
    else:
        eps_list = []
        eps_list.append(args.eps)
    print(eps_list)

    main(args.attack, args.src_model, args.target_model)
