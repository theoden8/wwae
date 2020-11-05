import numpy as np
import tensorflow as tf

import ops.linear
import ops.conv2d
import ops.deconv2d
import ops.batchnorm
import ops.layernorm
import ops._ops


#################################### Encoder/Decoder ####################################

######### mlp #########
def mlp_encoder(opts, input, output_dim, reuse=False,
                                        is_training=False):
    layer_x = input
    # hidden 0
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                1024, init=opts['mlp_init'], scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(
            opts, layer_x, 'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                1024, init=opts['mlp_init'], scope='hid1/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(
            opts, layer_x, 'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                1024, init=opts['mlp_init'], scope='hid2/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(
            opts, layer_x, 'hid2/bn', is_training, reuse)
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
                1024, init=opts['mlp_init'], scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(
            opts, layer_x, 'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                1024, init=opts['mlp_init'], scope='hid1/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(
            opts, layer_x, 'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                1024, init=opts['mlp_init'], scope='hid2/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(
            opts, layer_x, 'hid2/bn', is_training, reuse)
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
                                stride=3, scope='hid3/conv',
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
                                8*8*128, scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    layer_x = tf.reshape(layer_x, [-1, 8, 8, 128])
    # hidden 1
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                64]
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
                                32]
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
                                output_dim=64, filter_size=4,
                                stride=2, scope='hid0/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=128, filter_size=4,
                                stride=2, scope='hid1/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=256, filter_size=4,
                                stride=2, scope='hid2/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 3
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=512, filter_size=4,
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
                                128]
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
    # outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
    #                             output_shape=[batch_size,]+output_dim, filter_size=5,
    #                             stride=2, scope='hid_final/deconv',
    #                             init= opts['conv_init'])

    return outputs

net_archi = {'mlp': {'encoder': mlp_encoder, 'decoder': mlp_decoder},
            'mnist':{'encoder': mnist_conv_encoder, 'decoder': mnist_conv_decoder},
            'svhn':{'encoder': cifar10_conv_encoder, 'decoder': cifar10_conv_decoder},
            'cifar10':{'encoder': cifar10_conv_encoder, 'decoder': cifar10_conv_decoder},
            'celebA':{'encoder': celebA_conv_encoder, 'decoder': celebA_conv_decoder},
            }

#################################### Critic ####################################

######### mlp #########
def mlp_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    layer_x = tf.compat.v1.layers.flatten(inputs)
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    1024, init=opts['mlp_init'],
                                    scope='hid0/lin')
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    1024, init=opts['mlp_init'],
                                    scope='hid1/lin')
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 2
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    1024, init=opts['mlp_init'],
                                    scope='hid2/lin')
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 3
        layer_x = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    1024, init=opts['mlp_init'],
                                    scope='hid3/lin')
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    np.prod(in_shape),
                                    init=opts['mlp_init'],
                                    scope='hid_final')

    return outputs

######### conv #########
def conv_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
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
        # final layer
        outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    np.prod(in_shape),
                                    init=opts['mlp_init'],
                                    scope='hid_final')

    return outputs

######### fullconv #########
def cifar_fullconv_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    batch_size, in_shape =  tf.shape(inputs)[0], inputs.get_shape().as_list()[1:]
    layer_x = inputs
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=64, filter_size=4,
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

    return outputs

def celeba_fullconv_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    batch_size, in_shape =  tf.shape(inputs)[0], inputs.get_shape().as_list()[1:]
    layer_x = inputs
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=32, filter_size=5,
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
        _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                    2*layer_x.get_shape().as_list()[2],
                                    64]
        layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=_out_shape,
                                    filter_size=4,
                                    stride=2, scope='hid2/deconv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 3
        _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                    2*layer_x.get_shape().as_list()[2],
                                    32]
        layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=_out_shape,
                                    filter_size=5,
                                    stride=2, scope='hid3/deconv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=[batch_size,]+in_shape, filter_size=1,
                                    stride=1, scope='hid_final/deconv',
                                    init= opts['conv_init'])

    return outputs

critic_archi = {'mlp': mlp_critic,
            'conv': conv_critic,
            # 'fullconv': {'cifar10':cifar_fullconv_critic, 'celebA':celeba_fullconv_critic}
            'fullconv': {'cifar10':cifar_fullconv_critic, 'celebA':cifar_fullconv_critic}
            }
