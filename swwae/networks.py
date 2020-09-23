import numpy as np
import tensorflow as tf

import ops.linear
import ops.conv2d
import ops.deconv2d
import ops.batchnorm
import ops.layernorm
import ops._ops
from datahandler import datashapes
from sampling_functions import sample_gaussian
from net_archi import net_archi

import logging
import pdb

def encoder(opts, input, output_dim, scope=None,
                                    reuse=False,
                                    is_training=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        if opts['net_archi'] == 'mlp':
            encoder = net_archi['mlp']['encoder']
        elif opts['net_archi'] == 'small_conv':
            encoder = net_archi['small_conv']['encoder']
        elif opts['net_archi'] == 'conv':
            encoder = net_archi[opts['dataset']]['encoder']
        else:
            raise ValueError('Unknown {} net. archi.'.format(opts['net_archi']))
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
        elif opts['net_archi'] == 'small_conv':
            encoder = net_archi['small_conv']['decoder']
        elif opts['net_archi'] == 'conv':
            decoder = net_archi[opts['dataset']]['decoder']
        else:
            raise ValueError('Unknown {} dataset'.format(opts['net_archi']))
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
