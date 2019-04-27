import os
from os.path import basename
from os.path import join

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import keras.backend as K
import numpy as np
import tensorflow as tf
import scipy.misc
from tqdm import tqdm
# from es.draw_multiple import draw

from mnist import data_mnist, set_mnist_flags, load_model

CLIP_MIN = 0
CLIP_MAX = 1

ADVERSARIAL_DATA_PATH = '/Users/mchrusci/uj/shaper_data/adversarial'
RESULTS_PATH = '/Users/mchrusci/uj/shaper_data/adversarial/results.csv'


def class_means(X, y):
    """Return a list of means of each class in (X,y)"""

    classes = np.unique(y)
    no_of_classes = len(classes)
    means = []
    class_frac = []
    for item in classes:
        indices = np.where(y == item)[0]
        class_items = X[indices]
        class_frac.append(float(len(class_items)) / float(len(X)))
        mean = np.mean(class_items, axis=0)
        means.append(mean)
    return means, class_frac


def length_scales(X, y):
    """Find distances from each class mean to means of the other classes"""

    means, class_frac = class_means(X, y)
    no_of_classes = len(means)
    mean_dists = np.zeros((no_of_classes, no_of_classes))
    scales = []
    closest_means = np.zeros((no_of_classes))
    for i in range(no_of_classes):
        mean_diff = 0.0
        curr_mean = means[i]
        mean_not_i = 0.0
        curr_frac = class_frac[i]
        closest_mean = 1e6
        for j in range(no_of_classes):
            if i == j:
                mean_dists[i, j] = 0.0
            else:
                mean_dists[i, j] = np.linalg.norm(curr_mean - means[j])
                if mean_dists[i, j] < closest_mean:
                    closest_mean = mean_dists[i, j]
                    closest_means[i] = j
                mean_not_i = mean_not_i + means[j]

        mean_diff = curr_frac * curr_mean - (1 - curr_frac) * (mean_not_i / (no_of_classes - 1))
        scales.append(np.linalg.norm(mean_diff))
    return scales, mean_dists, closest_means


def naive_untargeted_attack(X, y):
    """
    Returns a minimum distance required to move a sample to a different class
    """

    scales = length_scales(X, y)
    print(scales)
    data_len = len(X)
    classes = np.unique(y)
    distances = []
    for i in range(100):
        curr_data = X[i, :]
        curr_distances = []
        for j in range(100):
            if i == j:
                continue
            else:
                if y[i] != y[j]:
                    data_diff = curr_data - X[j, :]
                    data_dist = np.linalg.norm(data_diff)
                    print(data_dist)
                    curr_distances.append(data_dist / scales[y[i]])
        distances.append(min(curr_distances))
    return distances


def main(target_model_name):
    redraw = False
    save_adv_samples_only = True

    np.random.seed(0)
    tf.set_random_seed(0)

    x = K.placeholder((None,
                       28,
                       28,
                       1))

    y = K.placeholder((None, 10))

    dim = int(28 * 28 * 1)

    _, _, X_test, Y_test = data_mnist()
    print('Loaded data')

    # target model for crafting adversarial examples
    target_model = load_model(target_model_name)
    target_model_name = basename(target_model_name)

    logits = target_model(x)
    prediction = K.softmax(logits)

    sess = tf.Session()
    print('Creating session')

    Y_test_uncat = np.argmax(Y_test, 1)

    means, class_frac = class_means(X_test, Y_test_uncat)

    scales, mean_dists, closest_means = length_scales(X_test, Y_test_uncat)

    eps_list = [args.eps]

    adv_images = np.empty((10 * 28, len(eps_list) * 28))

    total_X_adv = np.empty(X_test.shape)
    total_Y = np.empty(Y_test_uncat.shape)
    total_adv_pred = np.empty(Y_test_uncat.shape)
    total_adv_prob = np.empty(Y_test_uncat.shape)

    for eps_idx, eps in tqdm(enumerate(eps_list)):
        eps_orig = eps
        if args.alpha > eps:
            alpha = eps
            eps = 0
        elif eps >= args.alpha:
            alpha = args.alpha
            eps -= args.alpha

        adv_success = 0.0
        avg_l2_perturb = 0.0
        NUM_SAVED = 0
        for i in tqdm(range(10)):
            curr_indices = np.where(Y_test_uncat == i)
            NUM_SAMPLES = len(curr_indices[0])
            X_test_ini = X_test[curr_indices]
            Y_test_curr = Y_test_uncat[curr_indices]
            curr_len = len(X_test_ini)
            if args.targeted_flag == 1:
                allowed_targets = list(range(10))
                allowed_targets.remove(i)

            random_perturb = np.random.randn(*X_test_ini.shape)

            if args.norm == 'linf':
                random_perturb_signed = np.sign(random_perturb)
                X_test_curr = np.clip(X_test_ini + alpha * random_perturb_signed, CLIP_MIN,
                                      CLIP_MAX)
            elif args.norm == 'l2':
                random_perturb_unit = random_perturb / np.linalg.norm(
                    random_perturb.reshape(curr_len, dim), axis=1)[:,
                                                       None, None, None]
                X_test_curr = np.clip(X_test_ini + alpha * random_perturb_unit, CLIP_MIN, CLIP_MAX)

            if args.targeted_flag == 0:
                closest_class = int(closest_means[i])
                mean_diff_vec = means[closest_class] - means[i]
            elif args.targeted_flag == 1:
                targets = []
                mean_diff_array = np.zeros(
                    (curr_len, 28, 28, 1))
                for j in range(curr_len):
                    target = np.random.choice(allowed_targets)
                    targets.append(target)
                    mean_diff_array[j] = means[target] - means[i]

            if args.norm == 'linf':
                if args.targeted_flag == 0:
                    mean_diff_vec_signed = np.sign(mean_diff_vec)
                    perturb = eps * mean_diff_vec_signed
                elif args.targeted_flag == 1:
                    mean_diff_array_signed = np.sign(mean_diff_array)
                    perturb = eps * mean_diff_array_signed
            elif args.norm == 'l2':
                mean_diff_vec_unit = mean_diff_vec / np.linalg.norm(mean_diff_vec.reshape(dim))
                perturb = eps * mean_diff_vec_unit

            X_adv = np.clip(X_test_curr + perturb, CLIP_MIN, CLIP_MAX)

            assert X_adv.shape[1:] == total_X_adv.shape[1:], f'X_adv.shape[1:] = {X_adv.shape[
                                                                                  1:]}, total_X_adv.shape[1:] = {total_X_adv.shape[
                                                                                                                 1:]}'
            assert X_adv.shape[0] == NUM_SAMPLES, f'X_adv.shape[0] = {X_adv.shape[0]}'
            total_X_adv[NUM_SAVED:NUM_SAVED + NUM_SAMPLES] = X_adv

            # sample_x_adv = save_sample(X_adv, eps, i)

            if redraw:
                sample_x_adv = draw(
                    images=sample_x_adv.reshape(
                        (1, sample_x_adv.shape[0], sample_x_adv.shape[1], sample_x_adv.shape[2])),
                    n=10,
                    alpha=0.8,
                    background='000000'
                )[0]

            # row_start = i * 28
            # col_start = eps_idx * 28
            # no_channels_x = sample_x_adv.reshape(28, 28)
            # adv_images[row_start:row_start + 28,
            # col_start: col_start + 28] = no_channels_x

            # Getting the norm of the perturbation
            perturb_norm = np.linalg.norm((X_adv - X_test_ini).reshape(curr_len, dim), axis=1)
            perturb_norm_batch = np.mean(perturb_norm)
            avg_l2_perturb += perturb_norm_batch

            predictions_adv = \
                K.get_session().run([prediction], feed_dict={x: X_adv, K.learning_phase(): 0})[0]

            total_adv_pred[NUM_SAVED:NUM_SAVED + NUM_SAMPLES] = np.argmax(predictions_adv, axis=1)
            total_adv_prob[NUM_SAVED:NUM_SAVED + NUM_SAMPLES] = np.max(predictions_adv, axis=1)

            if args.targeted_flag == 0:
                adv_success += np.sum(np.argmax(predictions_adv, 1) != Y_test_curr)
                assert Y_test_curr.shape[1:] == total_Y.shape[
                                                1:], f'Y_test_curr.shape[1:] = {Y_test_curr.shape[
                                                                                1:]}, total_Y.shape[1:] = {total_Y.shape[
                                                                                                           1:]}'
                assert Y_test_curr.shape[0] == NUM_SAMPLES, f'Y_test_curr.shape[0] = {
                Y_test_curr.shape[0]}'
                total_Y[NUM_SAVED:NUM_SAVED + NUM_SAMPLES] = Y_test_curr
            elif args.targeted_flag == 1:
                targets_arr = np.array(targets)
                adv_success += np.sum(np.argmax(predictions_adv, 1) == targets_arr)
                assert targets_arr.shape[1:] == total_Y.shape[
                                                1:], f'targets_arr.shape[1:] = {targets_arr.shape[
                                                                                1:]}, total_Y.shape[1:] = {total_Y.shape[
                                                                                                           1:]}'
                assert targets_arr.shape[0] == NUM_SAMPLES, f'targets_arr.shape[0] = {
                targets_arr.shape[0]}'
                total_Y[NUM_SAVED:NUM_SAVED + NUM_SAMPLES] = targets_arr

            NUM_SAVED += NUM_SAMPLES

        err = 100.0 * adv_success / len(X_test)
        avg_l2_perturb = avg_l2_perturb / 10

        print(f'eps = {eps}, alpha = {alpha}, adv success = {err}')
        print(f'avg l2 pertub = {avg_l2_perturb}')

    if redraw:
        scipy.misc.imsave('baseline_attacks_redrawned.png', adv_images)
    else:
        # save_total(adv_images, avg_l2_perturb, err)
        file = os.path.join(ADVERSARIAL_DATA_PATH,
                            f'baseline-norm-{args.norm}-alpha-{args.alpha}-targeted-{args.targeted_flag}-adv-samples')
        np.savez(file=file, X=total_X_adv, Y=total_Y, pred=total_adv_pred, prob=total_adv_prob)


def save_total(adv_images, avg_l2_perturb, err):
    adv_images_name = f'baseline-norm-{args.norm}-alpha-{args.alpha}-targeted-{args.targeted_flag}-total.png'
    scipy.misc.imsave(join(ADVERSARIAL_DATA_PATH, adv_images_name), adv_images)
    with open(RESULTS_PATH, "a+") as csv:
        csv.write(
            f"baseline,{args.target_model},{args.norm},{args.alpha},{args.targeted_flag},{args.eps},{err},{avg_l2_perturb}\n")


def save_sample(X_adv, eps, i):
    # sample_x_idx = np.random.randint(0, X_adv.shape[0] + 1)
    sample_x_idx = 666
    sample_x_adv = X_adv[sample_x_idx]
    sample_x_adv_name = f'baseline-norm-{args.norm}-alpha-{args.alpha}-targeted-{args.targeted_flag}-eps-{eps}-i-{i}.png'
    sample_x_adv_path = join(ADVERSARIAL_DATA_PATH, sample_x_adv_name)
    scipy.misc.imsave(sample_x_adv_path,
                      sample_x_adv.reshape(28, 28))
    return sample_x_adv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("target_model", help="target model for attack")
    parser.add_argument("--norm", type=str, default='linf', help="Norm constraint to use")
    parser.add_argument("--alpha", type=float, default=0.6, help="Amount of randomness")
    parser.add_argument("--targeted_flag", type=int, default=0, help="Carry out targeted attack")
    parser.add_argument("--eps", type=float, default=0.3)

    args = parser.parse_args()

    set_mnist_flags()

    main(args.target_model)
