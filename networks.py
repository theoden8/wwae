import numpy as np
import tensorflow as tf
from math import pi

from datahandler import datashapes
from sampling_functions import sample_gaussian
from net_archi import net_archi , critic_archi

import logging
import pdb

def encoder(opts, input, output_dim, scope=None,
                                    reuse=False,
                                    is_training=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        if opts['net_archi'] == 'mlp':
            encoder = net_archi['mlp']['encoder']
        elif opts['net_archi'] == 'conv' or opts['net_archi'] == 'resnet':
            encoder = net_archi[opts['net_archi']][opts['dataset']]['encoder']
        else:
            raise ValueError('Unknown {} net. archi. for {} dataset'.format(opts['net_archi'],opts['dataset']))
        outputs = encoder(opts, input, output_dim,
                                    reuse,
                                    is_training)

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500)
    Sigma = tf.nn.softplus(logSigma)
    mean = tf.compat.v1.layers.flatten(mean)
    Sigma = tf.compat.v1.layers.flatten(Sigma)

    if opts['encoder'] == 'det':
        z = mean
    elif opts['encoder'] == 'gauss':
        qz_params = tf.concat((mean, Sigma), axis=-1)
        z = sample_gaussian(qz_params, 'tensorflow')
    else:
        assert False, 'Unknown encoder %s' % opts['encoder']

    return z, mean, Sigma


def decoder(opts, input, output_dim, scope=None,
                                    reuse=False,
                                    is_training=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        if opts['net_archi'] == 'mlp':
            decoder = net_archi['mlp']['decoder']
        elif opts['net_archi'] == 'conv' or opts['net_archi'] == 'resnet':
            decoder = net_archi[opts['net_archi']][opts['dataset']]['decoder']
        else:
            raise ValueError('Unknown {} net. archi. for {} dataset'.format(opts['net_archi'],opts['dataset']))
        outputs = decoder(opts, input, output_dim,
                                    reuse,
                                    is_training)

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500)
    Sigma = tf.nn.softplus(logSigma)

    mean = tf.compat.v1.layers.flatten(mean)
    Sigma = tf.compat.v1.layers.flatten(Sigma)
    # sampling from gaussian if needed
    if opts['decoder']=='gaussian':
        x = sample_gaussian(tf.tf.concat([mean,Sigma], axis=-1), typ='tensorflow')
    else:
        x = mean
    # normalise to [0,1] or [-1,1]
    if opts['input_normalize_sym']:
        x = tf.nn.tanh(x)
    else:
        x = tf.nn.sigmoid(x)

    x = tf.reshape(x, [-1] + datashapes[opts['dataset']])

    return x, mean, Sigma


def critic(opts, inputs, scope=None, is_training=False, reuse=False):
    """
    Critic network of the w1
    inputs: [batch,w,h,c]
    outputs: [batch,w,h,c]
    """
    in_shape = inputs.get_shape().as_list()[1:]
    if opts['wgan_critic_archi']=='coef':
        layer_x = inputs
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            coef = tf.compat.v1.get_variable("W", in_shape, tf.float32,
                    tf.random_normal_initializer(stddev=opts['init_std']))
            # element-wise multi
            outputs = layer_x*coef
    else:
        critic = critic_archi[opts['wgan_critic_archi']]
        outputs = critic(opts, inputs, scope, is_training, reuse)

    return tf.reshape(outputs, [-1,]+in_shape)
