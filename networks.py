import numpy as np
import tensorflow as tf
from math import ceil, sqrt

import ops.linear
import ops.conv2d
import ops.deconv2d
import ops.batchnorm
import ops.layernorm
import ops._ops
import ops.resnet
from datahandler import datashapes
from sampling_functions import sample_gaussian

import logging
import pdb

def encoder(opts, input, output_dim, scope=None,
                                        reuse=False,
                                        is_training=False,
                                        dropout_rate=1.):
    with tf.variable_scope(scope, reuse=reuse):
        if opts['network']['e_arch'] == 'mlp':
            # Encoder uses only fully connected layers with ReLUs
            outputs = mlp_encoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['e_arch'] == 'dcgan':
            # Fully convolutional architecture similar to DCGAN
            outputs = dcgan_encoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['e_arch'] == 'resnet':
            assert False, 'To Do'
            # Resnet archi similar to Improved training of WAGAN
            outputs = resnet_encoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        else:
            raise ValueError('%s : Unknown encoder architecture' % opts['network']['e_arch'])

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500)
    Sigma = tf.nn.softplus(logSigma)
    mean = tf.layers.flatten(mean)
    Sigma = tf.layers.flatten(Sigma)

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
                                        is_training=False,
                                        dropout_rate=1.):
    with tf.variable_scope(scope, reuse=reuse):
        if opts['network']['d_arch'] == 'mlp':
            # Encoder uses only fully connected layers with ReLUs
            outputs = mlp_decoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['d_arch'] == 'dcgan':
            # Fully convolutional architecture similar to DCGAN
            outputs = dcgan_decoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        elif opts['network']['d_arch'] == 'resnet':
            assert False, 'To Do'
            # Fully convolutional architecture similar to improve Wasserstein nGAN
            outputs = resnet_decoder(opts, input, output_dim,
                                            reuse,
                                            is_training,
                                            dropout_rate)
        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % opts['network']['d_arch'])

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500)
    Sigma = tf.nn.softplus(logSigma)

    mean = tf.layers.flatten(mean)
    Sigma = tf.layers.flatten(Sigma)

    if opts['input_normalize_sym']:
        x = tf.nn.tanh(mean)
    else:
        x = tf.nn.sigmoid(mean)

    x = tf.reshape(x, [-1] + datashapes[opts['dataset']])

    return x, mean, Sigma


def mlp_encoder(opts, input, output_dim, reuse=False,
                                        is_training=False,
                                        dropout_rate=1.):
    layer_x = input
    for i in range(opts['network']['e_nlayers']):
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                    opts['network']['e_nfilters'][i], init=opts['mlp_init'], scope='hid{}/lin'.format(i))
        # Note for mlp, batchnorm and layernorm are equivalent
        if opts['normalization']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['network']['e_nonlinearity'])
    outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, init=opts['mlp_init'], scope='hid_final')

    return outputs


def dcgan_encoder(opts, input, output_dim, reuse=False,
                                        is_training=False,
                                        dropout_rate=1.):
    """
    DCGAN style network with stride 2 at each hidden convolution layers.
    Final dense layer with output of size output_dim.
    """
    layer_x = input
    for i in range(opts['network']['e_nlayers']):
        layer_x = ops.conv2d.Conv2d(opts, layer_x,layer_x.get_shape().as_list()[-1],opts['network']['e_nfilters'][i],
                opts['network']['filter_size'][i],stride=2,scope='hid{}/conv'.format(i+1),init=opts['conv_init'])
        if opts['normalization']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
        layer_x = ops._ops.non_linear(layer_x,opts['network']['e_nonlinearity'])
    outputs = ops.linear.Linear(opts,layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, scope='hid_final')

    return outputs


def mlp_decoder(opts, input, output_dim, reuse=False,
                                        is_training=False,
                                        dropout_rate=1.):
    # Architecture with only fully connected layers and ReLUs
    layer_x = input
    for i in range(opts['network']['d_nlayers']):
        layer_x = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                    opts['network']['d_nfilters'][opts['network']['d_nlayers']-1-i], init=opts['mlp_init'], scope='hid%d/lin' % i)
        layer_x = ops._ops.non_linear(layer_x,opts['network']['d_nonlinearity'])
        # Note for mlp, batchnorm and layernorm are equivalent
        if opts['normalization']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % i, is_training, reuse)
    outputs = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                np.prod(output_dim), init=opts['mlp_init'], scope='hid_final')

    return outputs


def  dcgan_decoder(opts, input, output_dim, reuse,
                                            is_training,
                                            dropout_rate=1.):
    """
    DCGAN style network with stride 2 at each hidden deconvolution layers.
    First dense layer reshape to [out_h/2**num_layers,out_w/2**num_layers,num_units].
    Then num_layers deconvolutions with stride 2 and num_units filters.
    Last deconvolution output a 3-d latent code [out_h,out_w,2].
    """

    # batch_size
    batch_size = tf.shape(input)[0]
    # Linear layers
    height = ceil(output_dim[0] / 2**opts['network']['d_nlayers'])
    width = ceil(output_dim[1] / 2**opts['network']['d_nlayers'])
    h0 = input
    h0 = ops.linear.Linear(opts,h0,np.prod(h0.get_shape().as_list()[1:]),
                opts['network']['d_nfilters'][-1]*height*width, scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        h0 = ops.batchnorm.Batchnorm_layers(
            opts, h0, 'hid0/bn', is_training, reuse)
    h0 = ops._ops.non_linear(h0,'relu')
    h0 = tf.reshape(h0, [-1, height, width, opts['network']['d_nfilters'][-1]])
    layer_x = h0
    # Conv block
    for i in range(opts['network']['d_nlayers'] - 1):
        _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                        2*layer_x.get_shape().as_list()[2],
                        opts['network']['d_nfilters'][opts['network']['d_nlayers']-1-i]]
        layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], _out_shape,
                   opts['network']['filter_size'][opts['network']['d_nlayers']-1-i], stride=2, scope='hid%d/deconv' % i, init= opts['conv_init'])
        if opts['normalization']=='batchnorm':
            layer_x = ops.batchnorm.Batchnorm_layers(
                opts, layer_x, 'hid%d/bn' % (i+2), is_training, reuse)
        layer_x = ops._ops.non_linear(layer_x,'relu')
    outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1], [batch_size,]+output_dim,
                opts['network']['filter_size'][0], stride=2, scope='hid_final/deconv', init= opts['conv_init'])

    return outputs
