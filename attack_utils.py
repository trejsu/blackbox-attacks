import keras.backend as K
import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS


def linf_loss(X1, X2):
    return np.max(np.abs(X1 - X2), axis=(1, 2, 3))


def gen_adv_loss(logits, y, loss='logloss', mean=False):
    """
    Generate the loss function.
    """

    if loss == 'training':
        # use the model's output instead of the true labels to avoid
        # label leaking at training time
        y = K.cast(K.equal(logits, K.max(logits, 1, keepdims=True)), "float32")
        y = y / K.sum(y, 1, keepdims=True)
        out = K.categorical_crossentropy(logits, y, from_logits=True)
    elif loss == 'logloss':
        # out = K.categorical_crossentropy(logits, y, from_logits=True)
        out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        out = tf.reduce_mean(out)
    else:
        raise ValueError("Unknown loss: {}".format(loss))

    if mean:
        out = tf.mean(out)
    # else:
    #     out = K.sum(out)
    return out


def gen_grad(x, logits, y, loss='logloss'):
    """
    Generate the gradient of the loss function.
    """

    adv_loss = gen_adv_loss(logits, y, loss)

    # Define gradient of loss wrt input
    grad, = tf.gradients(adv_loss, x)
    return grad


def gen_hessian(x, logits, y, loss='logloss'):
    adv_loss = gen_adv_loss(logits, y, loss)

    hessian = tf.hessians(adv_loss, [x])[0]
    return hessian
