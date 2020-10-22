import tensorflow as tf

from networks import critic

import pdb

def wgan(opts, x, y, reuse=False):
    """
    Compute the W1 between images intensities.
    x[b,h,w,c]: true observation
    y[b,h,w,c]: reconstruction
    """

    # get images intensities
    x_int = x / tf.reduce_sum(x, axis=[1,2], keepdims=True) #[batch,w,h,c]
    y_int = y / tf.reduce_sum(y, axis=[1,2], keepdims=True) #[batch,w,h,c]
    # get pot.
    pot = critic(opts, x-y, scope='w1_critic', reuse=reuse) #[batch,w,h,c]
    # sum_diff
    cost = tf.reduce_sum(pot*(x_int-y_int), axis=[1,2]) #[batch,c]

    return tf.reduce_mean(cost, axis=-1)
