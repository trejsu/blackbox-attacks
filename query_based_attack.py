import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from os.path import basename

import keras.backend as K
import numpy as np
import scipy.misc
import tensorflow as tf
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from mnist import data_mnist, set_mnist_flags, load_model

from tqdm import tqdm

K.set_learning_phase(0)

FLAGS = tf.flags.FLAGS

RANDOM = True
BATCH_SIZE = 1
CLIP_MIN = 0
CLIP_MAX = 1
PARALLEL_FLAG = True

ADVERSARIAL_DATA_PATH = '/Users/mchrusci/uj/shaper_data/adversarial'
RESULTS_PATH = '/Users/mchrusci/uj/shaper_data/adversarial/results.csv'


def wb_write_out(eps, white_box_error, wb_norm):
    if RANDOM is False:
        print('Fraction of targets achieved (white-box) for {}: {}'.format(target, white_box_error))
    else:
        print('Fraction of targets achieved (white-box): {}'.format(white_box_error))
    return


def est_write_out(eps, success, avg_l2_perturb, X_adv=None):
    if RANDOM is False:
        print('Fraction of targets achieved (query-based) with {} for {}: {}'.format(
            target_model_name, target, success))
    else:
        print('Fraction of targets achieved (query-based): {}'.format(success))
    return


def pca_components(X, dim):
    X = X.reshape((len(X), dim))
    pca = PCA(n_components=dim)
    pca.fit(X)

    U = pca.components_.T
    U_norm = normalize(U, axis=0)

    return U_norm[:, :args.num_comp]


def xent_est(prediction, x, x_plus_i, x_minus_i, curr_target):
    pred_plus = run_model(prediction, x, x_plus_i)
    pred_plus_t = pred_plus[np.arange(BATCH_SIZE), list(curr_target)]
    pred_minus = run_model(prediction, x, x_minus_i)
    pred_minus_t = pred_minus[np.arange(BATCH_SIZE), list(curr_target)]
    single_grad_est = (pred_plus_t - pred_minus_t) / args.delta

    return single_grad_est / 2.0


def run_model(output_tensor, input_tensor, input_data):
    return K.get_session().run([output_tensor], feed_dict={input_tensor: input_data})[0]


def CW_est(logits, x, x_plus_i, x_minus_i, curr_sample, curr_target):
    curr_logits = run_model(logits, x, curr_sample)
    # So that when max is taken, it returns max among classes apart from the
    # target
    curr_logits[np.arange(BATCH_SIZE), list(curr_target)] = -1e4
    max_indices = np.argmax(curr_logits, 1)
    logit_plus = run_model(logits, x, x_plus_i)
    logit_plus_t = logit_plus[np.arange(BATCH_SIZE), list(curr_target)]
    logit_plus_max = logit_plus[np.arange(BATCH_SIZE), list(max_indices)]

    logit_minus = run_model(logits, x, x_minus_i)
    logit_minus_t = logit_minus[np.arange(BATCH_SIZE), list(curr_target)]
    logit_minus_max = logit_minus[np.arange(BATCH_SIZE), list(max_indices)]

    logit_t_grad_est = (logit_plus_t - logit_minus_t) / args.delta
    logit_max_grad_est = (logit_plus_max - logit_minus_max) / args.delta

    return logit_t_grad_est / 2.0, logit_max_grad_est / 2.0


def overall_grad_est(j, logits, prediction, x, curr_sample, curr_target,
    p_t, random_indices, num_groups, U=None):
    basis_vec = np.zeros((BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))

    if not PCA_FLAG:
        if j != num_groups - 1:
            curr_indices = random_indices[j * args.group_size:(j + 1) * args.group_size]
        elif j == num_groups - 1:
            curr_indices = random_indices[j * args.group_size:]
        row = curr_indices // FLAGS.IMAGE_COLS
        col = curr_indices % FLAGS.IMAGE_COLS
        for i in range(len(curr_indices)):
            basis_vec[:, row[i], col[i]] = 1.

    elif PCA_FLAG:
        basis_vec[:] = U[:, j].reshape((1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
        # basis_vec = np.sign(basis_vec)

    x_plus_i = np.clip(curr_sample + args.delta * basis_vec, CLIP_MIN, CLIP_MAX)
    x_minus_i = np.clip(curr_sample - args.delta * basis_vec, CLIP_MIN, CLIP_MAX)

    if args.loss_type == 'cw':
        logit_t_grad_est, logit_max_grad_est = CW_est(logits, x, x_plus_i,
                                                      x_minus_i, curr_sample, curr_target)
        if '_un' in args.method:
            single_grad_est = logit_t_grad_est - logit_max_grad_est
        else:
            single_grad_est = logit_max_grad_est - logit_t_grad_est
    elif args.loss_type == 'xent':
        single_grad_est = xent_est(prediction, x, x_plus_i, x_minus_i, curr_target)

    return single_grad_est


def finite_diff_method(prediction, logits, x, curr_sample, curr_target, p_t, dim, U=None):
    grad_est = np.zeros((BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS,
                         FLAGS.NUM_CHANNELS))
    logits_np = run_model(logits, x, curr_sample)
    if not PCA_FLAG:
        random_indices = np.random.permutation(dim)
        num_groups = dim // args.group_size
    elif PCA_FLAG:
        num_groups = args.num_comp
        random_indices = None

    if PARALLEL_FLAG:

        j_list = list(range(num_groups))

        # Creating partial function with single argument
        partial_overall_grad_est = partial(overall_grad_est, logits=logits,
                                           prediction=prediction, x=x, curr_sample=curr_sample,
                                           curr_target=curr_target, p_t=p_t,
                                           random_indices=random_indices, num_groups=num_groups,
                                           U=U)

        # Creating pool of threads
        pool = ThreadPool(3)
        all_grads = pool.map(partial_overall_grad_est, j_list)

        # print(len(all_grads))

        pool.close()
        pool.join()

        for j in j_list:
            # all_grads.append(partial_overall_grad_est(j))
            if not PCA_FLAG:
                if j != num_groups - 1:
                    curr_indices = random_indices[j * args.group_size:(j + 1) * args.group_size]
                elif j == num_groups - 1:
                    curr_indices = random_indices[j * args.group_size:]
                row = curr_indices // FLAGS.IMAGE_COLS
                col = curr_indices % FLAGS.IMAGE_COLS
            for i in range(len(curr_indices)):
                grad_est[:, row[i], col[i]] = all_grads[j].reshape((BATCH_SIZE, 1))

    else:
        for j in tqdm(range(num_groups)):
            single_grad_est = overall_grad_est(j, logits, prediction, x, curr_sample, curr_target,
                                               p_t, random_indices, num_groups, U)
            if not PCA_FLAG:
                if j != num_groups - 1:
                    curr_indices = random_indices[j * args.group_size:(j + 1) * args.group_size]
                elif j == num_groups - 1:
                    curr_indices = random_indices[j * args.group_size:]
                row = curr_indices // FLAGS.IMAGE_COLS
                col = curr_indices % FLAGS.IMAGE_COLS
                for i in range(len(curr_indices)):
                    grad_est[:, row[i], col[i]] = single_grad_est.reshape((BATCH_SIZE, 1))
            elif PCA_FLAG == True:
                basis_vec = np.zeros(
                    (BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
                basis_vec[:] = U[:, j].reshape(
                    (1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
                grad_est += basis_vec * single_grad_est[:, None, None, None]

    # Getting gradient of the loss
    if args.loss_type == 'xent':
        loss_grad = -1.0 * grad_est / p_t[:, None, None, None]
    elif args.loss_type == 'cw':
        logits_np_t = logits_np[np.arange(BATCH_SIZE), list(curr_target)].reshape(BATCH_SIZE)
        logits_np[np.arange(BATCH_SIZE), list(curr_target)] = -1e4
        max_indices = np.argmax(logits_np, 1)
        logits_np_max = logits_np[np.arange(BATCH_SIZE), list(max_indices)].reshape(BATCH_SIZE)
        logit_diff = logits_np_t - logits_np_max
        if '_un' in args.method:
            zero_indices = np.where(logit_diff + args.conf < 0.0)
        else:
            zero_indices = np.where(-logit_diff + args.conf < 0.0)
        grad_est[zero_indices[0]] = np.zeros(
            (len(zero_indices), FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
        loss_grad = grad_est

    return loss_grad


def estimated_grad_attack(X_test, X_test_ini, x, targets, prediction, logits, eps, dim, delta=None):
    success = 0
    avg_l2_perturb = 0
    time1 = time.time()
    U = None
    X_adv = np.zeros(
        (BATCH_SIZE * BATCH_EVAL_NUM, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
    if PCA_FLAG:
        U = pca_components(X_test, dim)

    total_Y = np.empty((X_adv.shape[0],))
    total_adv_pred = np.empty((X_adv.shape[0],))
    total_adv_prob = np.empty((X_adv.shape[0],))
    NUM_SAVED = 0

    for i in tqdm(range(BATCH_EVAL_NUM)):
        if i % 10 == 0:
            print('Batch no.: {}, {}'.format(i, eps))
        curr_sample = X_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].reshape(
            (BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, 1))
        curr_sample_ini = X_test_ini[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].reshape(
            (BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, 1))

        curr_target = targets[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        curr_prediction = run_model(prediction, x, curr_sample)

        p_t = curr_prediction[np.arange(BATCH_SIZE), list(curr_target)]

        if 'query_based' in args.method:
            loss_grad = finite_diff_method(prediction, logits, x, curr_sample,
                                           curr_target, p_t, dim, U)
        elif 'one_shot' in args.method:
            loss_grad = one_shot_method(prediction, x, curr_sample, curr_target, p_t)

        # Getting signed gradient of loss
        if args.norm == 'linf':
            normed_loss_grad = np.sign(loss_grad)
        elif args.norm == 'l2':
            grad_norm = np.linalg.norm(loss_grad.reshape(BATCH_SIZE, dim), axis=1)
            indices = np.where(grad_norm != 0.0)
            normed_loss_grad = np.zeros_like(curr_sample)
            normed_loss_grad[indices] = loss_grad[indices] / grad_norm[indices, None, None, None]

        eps_mod = eps - args.alpha
        if args.loss_type == 'xent':
            if '_un' in args.method:
                x_adv = np.clip(curr_sample + eps_mod * normed_loss_grad, 0, 1)
            else:
                x_adv = np.clip(curr_sample - eps_mod * normed_loss_grad, 0, 1)
        elif args.loss_type == 'cw':
            x_adv = np.clip(curr_sample - eps_mod * normed_loss_grad, 0, 1)

        # save_single_sample(X_adv, eps, i)

        # Getting the norm of the perturbation
        perturb_norm = np.linalg.norm((x_adv - curr_sample_ini).reshape(BATCH_SIZE, dim), axis=1)
        X_adv[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = x_adv.reshape(
            (BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, 1))
        perturb_norm_batch = np.mean(perturb_norm)
        avg_l2_perturb += perturb_norm_batch

        adv_prediction = run_model(prediction, x, x_adv)
        success += np.sum(np.argmax(adv_prediction, 1) == curr_target)

        total_adv_pred[NUM_SAVED:NUM_SAVED + BATCH_SIZE] = np.argmax(adv_prediction, axis=1)
        total_adv_prob[NUM_SAVED:NUM_SAVED + BATCH_SIZE] = np.max(adv_prediction, axis=1)
        total_Y[NUM_SAVED:NUM_SAVED + BATCH_SIZE] = curr_target

        NUM_SAVED += BATCH_SIZE

    success = 100.0 * float(success) / (BATCH_SIZE * BATCH_EVAL_NUM)

    if '_un' in args.method:
        success = 100.0 - success

    avg_l2_perturb = avg_l2_perturb / BATCH_EVAL_NUM

    est_write_out(eps, success, avg_l2_perturb, X_adv)

    time2 = time.time()
    print('Average l2 perturbation: {}'.format(avg_l2_perturb))
    print('Total time: {}, Average time: {}'.format(time2 - time1, (time2 - time1) / (
        BATCH_SIZE * BATCH_EVAL_NUM)))

    save_total(eps, X_adv, success, avg_l2_perturb, total_Y, total_adv_pred, total_adv_prob)

    return


def save_single_sample(X_adv, eps, i):
    # sample_x_idx = np.random.randint(0, X_adv.shape[0] + 1)
    sample_x_idx = 666
    sample_x_adv = X_adv[sample_x_idx]
    sample_x_adv_name = f'{args.method}-norm-{args.norm}-eps-{eps}-i-{i}.png'
    sample_x_adv_path = os.path.join(ADVERSARIAL_DATA_PATH, sample_x_adv_name)
    scipy.misc.imsave(sample_x_adv_path,
                      sample_x_adv.reshape(FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS))


def save_total(eps, adv_images, err, avg_l2_perturb, Y, pred, prob):
    file = os.path.join(ADVERSARIAL_DATA_PATH,
                        f'{args.method}-norm-{args.norm}-eps-{eps}-loss-{args.loss_type}-adv-samples')
    np.savez(file=file, X=adv_images, Y=Y, pred=pred, prob=prob)
    targeted = '_un' not in args.method
    with open(RESULTS_PATH, "a+") as csv:
        csv.write(
            f"{args.method},{args.target_model},{args.norm},{args.alpha},{targeted},{eps},{err},{avg_l2_perturb}\n")


def estimated_grad_attack_iter(X_test, X_test_ini, x, targets, prediction, logits, eps, dim, beta):
    success = 0
    avg_l2_perturb = 0
    time1 = time.time()
    U = None
    X_adv = np.zeros(
        (BATCH_SIZE * BATCH_EVAL_NUM, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
    if PCA_FLAG == True:
        U = pca_components(X_test, dim)


    total_Y = np.empty((X_adv.shape[0],))
    total_adv_pred = np.empty((X_adv.shape[0],))
    total_adv_prob = np.empty((X_adv.shape[0],))
    NUM_SAVED = 0

    for i in tqdm(range(BATCH_EVAL_NUM)):
        if i % 10 == 0:
            print('Batch no.: {}, {}'.format(i, eps))
        curr_sample = X_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].reshape(
            (BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, 1))
        curr_sample_ini = X_test_ini[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].reshape(
            (BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, 1))

        curr_target = targets[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        eps_mod = eps - args.alpha

        for j in tqdm(range(args.num_iter)):
            if j % 10 == 0:
                print('Num_iter:{}'.format(j))
            curr_prediction = run_model(prediction, x, curr_sample)

            p_t = curr_prediction[np.arange(BATCH_SIZE), list(curr_target)]

            if 'query_based' in args.method:
                loss_grad = finite_diff_method(prediction, logits, x, curr_sample,
                                               curr_target, p_t, dim, U)
            elif 'spsa' in args.method:
                loss_grad = spsa(prediction, logits, x, curr_sample,
                                 curr_target, p_t, dim)
                # print loss_grad.shape

            # Getting signed gradient of loss
            if args.norm == 'linf':
                normed_loss_grad = np.sign(loss_grad)
            elif args.norm == 'l2':
                grad_norm = np.linalg.norm(loss_grad.reshape(BATCH_SIZE, dim), axis=1)
                indices = np.where(grad_norm != 0.0)
                normed_loss_grad = np.zeros_like(curr_sample)
                normed_loss_grad[indices] = loss_grad[indices] / grad_norm[
                    indices, None, None, None]

            if args.loss_type == 'xent':
                if '_un' in args.method:
                    x_adv = np.clip(curr_sample + beta * normed_loss_grad, 0, 1)
                else:
                    x_adv = np.clip(curr_sample - beta * normed_loss_grad, 0, 1)
            elif args.loss_type == 'cw':
                x_adv = np.clip(curr_sample - beta * normed_loss_grad, 0, 1)
            r = x_adv - curr_sample_ini
            r = np.clip(r, -eps, eps)
            curr_sample = curr_sample_ini + r

            logits_curr = run_model(logits, x, curr_sample)
            logits_curr_t = logits_curr[np.arange(BATCH_SIZE), list(curr_target)].reshape(
                BATCH_SIZE)
            logits_curr[np.arange(BATCH_SIZE), list(curr_target)] = -1e4
            max_indices = np.argmax(logits_curr, 1)
            logits_curr_max = logits_curr[np.arange(BATCH_SIZE), list(max_indices)].reshape(
                BATCH_SIZE)
            loss = logits_curr_t - logits_curr_max
            # print loss

        x_adv = np.clip(curr_sample, 0, 1)

        # save_single_sample(X_adv, eps, i)

        # Getting the norm of the perturbation
        perturb_norm = np.linalg.norm((x_adv - curr_sample_ini).reshape(BATCH_SIZE, dim), axis=1)
        perturb_norm_batch = np.mean(perturb_norm)
        avg_l2_perturb += perturb_norm_batch

        adv_prediction = run_model(prediction, x, x_adv)
        X_adv[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = x_adv.reshape(
            (BATCH_SIZE, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, 1))
        success += np.sum(np.argmax(adv_prediction, 1) == curr_target)

        total_adv_pred[NUM_SAVED:NUM_SAVED + BATCH_SIZE] = np.argmax(adv_prediction, axis=1)
        total_adv_prob[NUM_SAVED:NUM_SAVED + BATCH_SIZE] = np.max(adv_prediction, axis=1)
        total_Y[NUM_SAVED:NUM_SAVED + BATCH_SIZE] = curr_target

        NUM_SAVED += BATCH_SIZE

    success = 100.0 * float(success) / (BATCH_SIZE * BATCH_EVAL_NUM)

    if '_un' in args.method:
        success = 100.0 - success

    avg_l2_perturb = avg_l2_perturb / BATCH_EVAL_NUM

    est_write_out(eps, success, avg_l2_perturb, X_adv)

    time2 = time.time()
    print('Average l2 perturbation: {}'.format(avg_l2_perturb))
    print('Total time: {}, Average time: {}'.format(time2 - time1, (time2 - time1) / (
        BATCH_SIZE * BATCH_EVAL_NUM)))

    save_total(eps, X_adv, success, avg_l2_perturb, total_Y, total_adv_pred, total_adv_prob)

    return


def main(target_model_name, target=None):
    np.random.seed(0)
    tf.set_random_seed(0)

    x = K.placeholder((None,
                       FLAGS.IMAGE_ROWS,
                       FLAGS.IMAGE_COLS,
                       FLAGS.NUM_CHANNELS))

    y = K.placeholder((None, FLAGS.NUM_CLASSES))

    dim = int(FLAGS.IMAGE_ROWS * FLAGS.IMAGE_COLS)

    _, _, X_test_ini, Y_test = data_mnist()
    print('Loaded data')

    Y_test_uncat = np.argmax(Y_test, axis=1)

    # target model for crafting adversarial examples
    target_model = load_model(target_model_name)
    target_model_name = basename(target_model_name)

    logits = target_model(x)
    prediction = K.softmax(logits)



    print('Creating session')

    if '_un' in args.method:
        targets = np.argmax(Y_test[:BATCH_SIZE * BATCH_EVAL_NUM], 1)
    elif RANDOM is False:
        targets = np.array([target] * (BATCH_SIZE * BATCH_EVAL_NUM))
    elif RANDOM is True:
        targets = []
        allowed_targets = list(range(FLAGS.NUM_CLASSES))
        for i in range(BATCH_SIZE * BATCH_EVAL_NUM):
            allowed_targets.remove(Y_test_uncat[i])
            targets.append(np.random.choice(allowed_targets))
            allowed_targets = list(range(FLAGS.NUM_CLASSES))
        # targets = np.random.randint(10, size = BATCH_SIZE*BATCH_EVAL_NUM)
        targets = np.array(targets)
        # print(targets)
    targets_cat = np_utils.to_categorical(targets, FLAGS.NUM_CLASSES).astype(np.float32)

    if args.norm == 'linf':
        # eps_list = list(np.linspace(0.025, 0.1, 4))
        # eps_list.extend(np.linspace(0.15, 0.5, 8))
        eps_list = [0.3]
        if "_iter" in args.method:
            eps_list = [0.3]
    elif args.norm == 'l2':
        eps_list = list(np.linspace(0.0, 2.0, 5))
        eps_list.extend(np.linspace(2.5, 9.0, 14))
        # eps_list = [5.0]
    # print(eps_list)

    adv_images = np.empty((FLAGS.NUM_CLASSES * FLAGS.IMAGE_ROWS, len(eps_list) * FLAGS.IMAGE_COLS))

    random_perturb = np.random.randn(*X_test_ini.shape)

    if args.norm == 'linf':
        random_perturb_signed = np.sign(random_perturb)
        X_test = np.clip(X_test_ini + args.alpha * random_perturb_signed, CLIP_MIN, CLIP_MAX)
    elif args.norm == 'l2':
        random_perturb_unit = random_perturb / np.linalg.norm(random_perturb.reshape(curr_len, dim),
                                                              axis=1)[:, None, None, None]
        X_test = np.clip(X_test_ini + args.alpha * random_perturb_unit, CLIP_MIN, CLIP_MAX)

    for eps in eps_list:
        if '_iter' in args.method:
            # white_box_fgsm_iter(prediction, target_model, x, logits, y, X_test, X_test_ini, targets,
            #                     targets_cat, eps, dim, args.beta)
            estimated_grad_attack_iter(X_test, X_test_ini, x, targets, prediction, logits, eps, dim,
                                       args.beta)
        else:
            # white_box_fgsm(prediction, target_model, x, logits, y, X_test, X_test_ini, targets,
            #                targets_cat, eps, dim)
            estimated_grad_attack(X_test, X_test_ini, x, targets, prediction, logits, eps, dim)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("target_model", help="target model for attack")
    parser.add_argument("--method", choices=['query_based', 'spsa_iter',
                                             'query_based_un', 'spsa_un_iter',
                                             'query_based_un_iter', 'query_based_iter'],
                        default='query_based_un')
    parser.add_argument("--delta", type=float, default=0.01,
                        help="local perturbation")
    parser.add_argument("--norm", type=str, default='linf',
                        help="Norm to use for attack")
    parser.add_argument("--loss_type", type=str, default='xent',
                        help="Choosing which type of loss to use")
    parser.add_argument("--conf", type=float, default=0.0,
                        help="Strength of CW sample")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Strength of random perturbation")
    parser.add_argument("--group_size", type=int, default=1,
                        help="Number of features to group together")
    parser.add_argument("--num_comp", type=int, default=784,
                        help="Number of pca components")
    parser.add_argument("--num_iter", type=int, default=40,
                        help="Number of iterations")
    parser.add_argument("--beta", type=int, default=0.01,
                        help="Step size per iteration")

    args = parser.parse_args()

    target_model_name = basename(args.target_model)

    set_mnist_flags()

    if '_un' in args.method:
        RANDOM = True
    PCA_FLAG = False
    if args.num_comp != 784:
        PCA_FLAG = True

    if '_iter' in args.method:
        BATCH_EVAL_NUM = 10
    else:
        BATCH_EVAL_NUM = 10

    if RANDOM is False:
        for i in range(FLAGS.NUM_CLASSES):
            main(args.target_model, i)
    elif RANDOM is True:
        main(args.target_model)
