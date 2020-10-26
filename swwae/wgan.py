import tensorflow as tf

from networks import critic

import pdb

def wgan(opts, x, y, reuse=False):
    """
    Compute the W1 between images intensities.
    x[b,h,w,c]: true observation
    y[b,h,w,c]: reconstruction
    """
    h, w, c = x.get_shape().as_list()[1:]
    # get per channel masses
    mx = tf.reduce_sum(x, axis=[1,2], keepdims=True)
    my = tf.reduce_sum(y, axis=[1,2], keepdims=True)
    # get images intensities
    x_int = x / mx #[batch,w,h,c]
    y_int = y / my #[batch,w,h,c]
    # get pot.
    critic_ouput = critic(opts, x-y, scope='w1_critic', reuse=reuse) #[batch,w,h,c]
    # sum_diff
    cost = tf.reduce_sum(critic_ouput*(x_int-y_int), axis=[1,2]) #[batch,c]
    cost += opts['gamma'] * (1. - tf.reshape(my/mx,[-1,c]))**2
    # critic Lips. reg
    reg = critic_reg(critic_ouput) #[batch,c]

    return tf.reduce_mean(cost, axis=-1), tf.reduce_mean(reg)

def critic_reg(critic_ouput):
    """
    Compute lipschitz reg for the critic: (|f_ij - f_kl|-|ij-kl|_l2)^2
    critic_ouput[b,h,w,c]: output of the critic.
    """
    losses = []
    for i in range(3):
        for j in range(3):
            if i!=1 and j!=1:
                padding = [[0,0], [i,2-i], [j,2-j], [0,0]]
                grad = tf.pad(critic_ouput, padding)-tf.pad(critic_ouput, [[0,0], [1,1], [1,1], [0,0]])
                if i==j:
                    l = tf.reduce_sum(tf.square(tf.abs(grad[:,1:-1,1:-1])-tf.sqrt(2.)), axis=[1,2])
                else:
                    l = tf.reduce_sum(tf.square(tf.abs(grad[:,1:-1,1:-1])-1), axis=[1,2])
                losses.append(l)

    return tf.reduce_sum(tf.stack(losses, axis=-1), axis=-1)
