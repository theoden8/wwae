import tensorflow as tf

from networks import critic

import pdb

def wgan(opts, x, y, is_training=False, reuse=False):
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
    critic_ouput = critic(opts, x-y, 'w1_critic', is_training, reuse) #[batch,w,h,c]
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
    glob_max_grad = 0.
    for i in range(3):
        for j in range(3):
            if i!=1 and j!=1:
                padding = [[0,0], [i,2-i], [j,2-j], [0,0]]
                grad = tf.pad(critic_ouput, padding, mode='SYMMETRIC')-tf.pad(critic_ouput, [[0,0], [1,1], [1,1], [0,0]], mode='SYMMETRIC')
                max_grad = tf.reduce_max(tf.abs(grad[:,1:-1,1:-1]), axis=[1,2])
                if i==j or (i==2 and j==0) or (i==0 and j==2):
                    # update max_grad_diag if needed
                    glob_max_grad = tf.maximum(max_grad / tf.sqrt(2.), glob_max_grad)
                else:
                    # update max_grad_side if needed
                    glob_max_grad = tf.maximum(max_grad, glob_max_grad)

    return tf.square(glob_max_grad-1)
