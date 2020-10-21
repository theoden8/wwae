import tensorflow as tf

from networks import critic

import pdb

def wgan(opts, x, y, reuse=False):
    """
    Compute the W1 between images intensities.
    x[b,h,w,c]: true observation
    y[b,h,w,c]: reconstruction
    """

    # stopping gradient from decoder
    rec = tf.stop_gradient(y)
    # get pot.
    pot_obs = get_pot(opts,x,reuse=reuse)
    pot_rec = get_pot(opts,rec,reuse=True)
    diff = pot_obs-pot_rec

    return tf.reduce_mean(diff, axis=-1)

def get_pot(opts,x,reuse=False):
    """ Wrapper to get weighted potential sum_ij f_ij x_ij
    """
    # normalize images to get intensities
    x_norm = x / tf.reduce_sum(x, axis=[1,2], keepdims=True) #[batch,w,h,c]
    # get critic of inputs
    x_critic = critic(opts, x, scope='w1_critic', reuse=reuse) #[batch,w,h,c]
    # sum over pixel space
    sum = tf.reduce_sum(x_norm*x_critic, axis=[1,2]) #[batch,c]

    return sum
