import numpy as np
import tensorflow as tf
from math import pi

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

def theta_discriminator(opts, inputs, output_dim,
                                    scope=None,
                                    reuse=False):
    """
    Discriminator network to learn proj. dir.
    inputs: [batch,w,h,c]
    outputs: [batch,L*output_dim]
    """
    in_shape = inputs.get_shape().as_list()[1:]
    layer_x = tf.compat.v1.layers.flatten(inputs)
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    512, init=opts['mlp_init'],
                                    scope='hid0/lin')
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    512, init=opts['mlp_init'],
                                    scope='hid1/lin')
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # # hidden 2
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    512, init=opts['mlp_init'],
                                    scope='hid2/lin')
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    opts['sw_proj_num']*output_dim,
                                    init=opts['mlp_init'],
                                    scope='hid_final')
        if opts['sw_proj_type']=='max-sw':
            # rescaling theta between -pi/2 and pi/2
            outputs = tf.math.minimum(pi/2., tf.math.maximum(-pi/2., outputs))
        elif opts['sw_proj_type']=='max-gsw':
            # proj between -1 and 1
            outputs = tf.nn.tanh(outputs)
        else:
            raise ValueError('Unknown {} sw projection' % opts['sw_proj_type'])

    return outputs

def critic(opts, inputs, scope=None, reuse=False):
    """
    Critic network of the w1
    inputs: [batch,w,h,c]
    outputs: [batch,w,h,c]
    """
    batch_size, in_shape =  tf.shape(inputs)[0], inputs.get_shape().as_list()[1:]
    if opts['wgan_critic_archi']=='mlp':
        layer_x = tf.compat.v1.layers.flatten(inputs)
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # hidden 0
            layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                        512, init=opts['mlp_init'],
                                        scope='hid0/lin')
            layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
            # hidden 1
            layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                        512, init=opts['mlp_init'],
                                        scope='hid1/lin')
            layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
            # hidden 2
            layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                        512, init=opts['mlp_init'],
                                        scope='hid2/lin')
            layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
            # final layer
            outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                        np.prod(in_shape),
                                        init=opts['mlp_init'],
                                        scope='hid_final')
    elif opts['wgan_critic_archi']=='conv':
        layer_x = inputs
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # hidden 0
            layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_dim=32, filter_size=4,
                                        stride=2, scope='hid0/conv',
                                        init=opts['conv_init'])
            layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
            # hidden 1
            layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_dim=64, filter_size=4,
                                        stride=2, scope='hid1/conv',
                                        init=opts['conv_init'])
            layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
            # hidden 2
            layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_dim=128, filter_size=4,
                                        stride=2, scope='hid2/conv',
                                        init=opts['conv_init'])
            layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
            # final layer
            outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                        np.prod(in_shape),
                                        init=opts['mlp_init'],
                                        scope='hid_final')
    elif opts['wgan_critic_archi']=='fullconv':
        layer_x = inputs
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # hidden 0
            layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_dim=32, filter_size=4,
                                        stride=2, scope='hid0/conv',
                                        init=opts['conv_init'])
            layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
            # hidden 1
            layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_dim=64, filter_size=4,
                                        stride=2, scope='hid1/conv',
                                        init=opts['conv_init'])
            layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
            # hidden 2
            layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_shape=[batch_size,int(in_shape[0]/2),int(in_shape[1]/2),32],
                                        filter_size=4,
                                        stride=2, scope='hid2/conv',
                                        init=opts['conv_init'])
            layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
            # final layer
            outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_shape=[batch_size,]+in_shape, filter_size=4,
                                        stride=2, scope='hid_final/deconv',
                                        init= opts['conv_init'])
    elif opts['wgan_critic_archi']=='shallowfullconv':
        layer_x = inputs
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # hidden 0
            layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_dim=32, filter_size=4,
                                        stride=2, scope='hid0/conv',
                                        init=opts['conv_init'])
            layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
            # hidden 1
            layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_shape=[batch_size,in_shape[0],in_shape[1],32],
                                        filter_size=4,
                                        stride=2, scope='hid1/deconv',
                                        init=opts['conv_init'])
            layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
            # final layer
            outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_shape=[batch_size,]+in_shape, filter_size=1,
                                        stride=1, scope='hid_final/deconv',
                                        init= opts['conv_init'])
    else:
        raise ValueError('Unknown {} archi for critic' % opts['wgan_critic_archi'])
    # clipping outputs
    if opts['wgan_critic_clip']=='piecewise':
        outputs = tf.math.minimum(.5, tf.math.maximum(-.5, outputs))
    elif opts['wgan_critic_clip']=='tanh':
        outputs = .5*tf.nn.tanh(outputs)
    # else:
    #     raise ValueError('Unknown {} clipping' % opts['wgan_critic_clip'])

    return tf.reshape(outputs, [-1,]+in_shape)

def obs_discriminator(opts, inputs, scope=None,
                                    reuse=False):
    """
    Discriminator network to transform RGB images
    inputs: [batch,w,h,c]
    outputs: [batch,w,h,1]
    """
    in_shape = inputs.get_shape().as_list()[1:]
    layer_x = tf.compat.v1.layers.flatten(inputs)
    with tf.compat.v1.variable_scope(scope, reuse=reuse):

        outputs = ops.linear.Linear(opts, layer_x,
                                    np.prod(in_shape),
                                    np.prod(in_shape[:-1]),
                                    init=opts['mlp_init'],
                                    scope='hid_final')

    return tf.reshape(tf.nn.sigmoid(outputs),[-1,in_shape[0],in_shape[1],1])
