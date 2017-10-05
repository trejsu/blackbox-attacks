import sys

import numpy as np
import tensorflow as tf
import tqdm

import models

# usage: python generate_test_fgsm_targeted.py <destination> <ckpt_dir> <epsilon>

dest = sys.argv[1]
ckpt_dir = sys.argv[2]
epsilon = float(sys.argv[3])

images_array = np.load('test_orig.npy')
targets_array = np.load('test_random_targets.npy')
images_batches = images_array.reshape((100, 100, 32, 32, 3))
targets_batches = targets_array.reshape((100, 100, 10))

images = tf.placeholder(shape=(100, 32, 32, 3), dtype=tf.float32)
labels = tf.placeholder(shape=(100, 10), dtype=tf.float32)
net = models.load_model(ckpt_dir, 100, images, labels)
loss = -net.get_loss() # for targeted, use minus loss and pass targets as labels
grads, = tf.gradients(loss, images, name='gradients_fgsm')
perturbation = epsilon * tf.sign(grads)
adv_images = tf.stop_gradient(tf.clip_by_value(images + perturbation, 0., 255.))

sess = tf.Session()
net.load(sess)

adv_images_array = np.zeros((10000, 32, 32, 3), dtype=np.float32)
adv_images_batches = adv_images_array.reshape((100, 100, 32, 32, 3))
for i in tqdm.trange(100):
    adv_images_batches[i] = sess.run(adv_images, feed_dict={
        images: images_batches[i],
        labels: targets_batches[i],
    })

np.save(dest, adv_images_array)