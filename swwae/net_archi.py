import numpy as np
import tensorflow as tf

import ops.linear
import ops.conv2d
import ops.deconv2d
import ops.batchnorm
import ops.layernorm
import ops._ops


######### mlp #########
def mlp_encoder(opts, input, output_dim, reuse=False,
                                        is_training=False):
    layer_x = input
    # hidden 0
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                512, init=opts['mlp_init'], scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(
            opts, layer_x, 'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                512, init=opts['mlp_init'], scope='hid1/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(
            opts, layer_x, 'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim, init=opts['mlp_init'], scope='hid_final')

    return outputs

def mlp_decoder(opts, input, output_dim, reuse=False,
                                        is_training=False):
    layer_x = input
    # hidden 0
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                512, init=opts['mlp_init'], scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(
            opts, layer_x, 'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                512, init=opts['mlp_init'], scope='hid1/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(
            opts, layer_x, 'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = ops.linear.Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                np.prod(output_dim), init=opts['mlp_init'], scope='hid_final')

    return outputs


######### mnist/svhn #########
def mnist_conv_encoder(opts, input, output_dim, reuse=False,
                                            is_training=False):
    """
    Archi used by Ghosh & al.
    """
    layer_x = input
    # hidden 0
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=32, filter_size=4,
                                stride=2, scope='hid0/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=64, filter_size=4,
                                stride=2, scope='hid1/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=128, filter_size=4,
                                stride=2, scope='hid2/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 3
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=256, filter_size=4,
                                stride=2, scope='hid3/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid3/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                output_dim, scope='hid_final')

    return outputs

def  mnist_conv_decoder(opts, input, output_dim, reuse,
                                            is_training):
    """
    Archi used by Ghosh & al.
    """
    # batch_size
    batch_size = tf.shape(input)[0]
    layer_x = input
    # Linear layers
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                8*8*256, scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    layer_x = tf.reshape(layer_x, [-1, 8, 8, 256])
    # hidden 1
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                128]
    layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid1/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers( opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                64]
    layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid2/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers( opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=[batch_size,]+output_dim, filter_size=1,
                                stride=1, scope='hid_final/deconv',
                                init= opts['conv_init'])

    return outputs


######### cifar10 #########
def cifar10_conv_encoder(opts, input, output_dim, reuse=False,
                                            is_training=False):
    """
    Archi used by Ghosh & al.
    """
    layer_x = input
    # hidden 0
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=128, filter_size=4,
                                stride=2, scope='hid0/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=256, filter_size=4,
                                stride=2, scope='hid1/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=512, filter_size=4,
                                stride=2, scope='hid2/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 3
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=1048, filter_size=4,
                                stride=2, scope='hid3/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid3/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                output_dim, scope='hid_final')

    return outputs

def  cifar10_conv_decoder(opts, input, output_dim, reuse,
                                            is_training):
    """
    Archi used by Ghosh & al.
    """
    # batch_size
    batch_size = tf.shape(input)[0]
    layer_x = input
    # Linear layers
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                8*8*1048, scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    layer_x = tf.reshape(layer_x, [-1, 8, 8, 1048])
    # hidden 1
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                512]
    layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid1/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers( opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                256]
    layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid2/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers( opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=[batch_size,]+output_dim, filter_size=1,
                                stride=1, scope='hid_final/deconv',
                                init= opts['conv_init'])

    return outputs


######### celebA #########
def celebA_conv_encoder(opts, input, output_dim, reuse=False,
                                            is_training=False):
    """
    Archi used by Ghosh & al.
    """
    layer_x = input
    # hidden 0
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=64, filter_size=5,
                                stride=2, scope='hid0/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=128, filter_size=5,
                                stride=2, scope='hid1/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=256, filter_size=5,
                                stride=2, scope='hid2/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 3
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=512, filter_size=5,
                                stride=2, scope='hid3/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid3/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                output_dim, scope='hid_final')

    return outputs

def  celebA_conv_decoder(opts, input, output_dim, reuse,
                                            is_training):
    """
    Archi used by Ghosh & al.
    """
    # batch_size
    batch_size = tf.shape(input)[0]
    layer_x = input
    # Linear layers
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                8*8*512, scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    layer_x = tf.reshape(layer_x, [-1, 8, 8, 512])
    # hidden 1
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                256]
    layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=5,
                                stride=2, scope='hid1/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers( opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                128]
    layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=5,
                                stride=2, scope='hid2/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers( opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 3
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                64]
    layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=5,
                                stride=2, scope='hid3/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers( opts, layer_x,
                                'hid3/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=[batch_size,]+output_dim, filter_size=1,
                                stride=1, scope='hid_final/deconv',
                                init= opts['conv_init'])

    return outputs

net_archi = {'mlp': {'encoder': mlp_encoder, 'decoder': mlp_decoder},
            'dcgan':{'encoder': dcgan_encoder, 'decoder': dcgan_decoder},
            'celeba_dcgan':{'encoder': dcgan_encoder, 'decoder': celebA_dcgan_decoder},
            'mnist':{'encoder': mnist_conv_encoder, 'decoder': mnist_conv_decoder},
            'svhn':{'encoder': mnist_conv_encoder, 'decoder': mnist_conv_decoder},
            'cifar10':{'encoder': cifar10_conv_encoder, 'decoder': cifar10_conv_decoder},
            'celebA':{'encoder': celebA_conv_encoder, 'decoder': celebA_conv_decoder},
            }
