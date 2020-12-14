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
    w,h = output_dim[0], output_dim[1]
    layer_x = input
    # Linear layers
    _out_shape = [int(w/2**3), int(h/2**3), 512]
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                np.prod(_out_shape), scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    layer_x = tf.reshape(layer_x, [-1,] + _out_shape)
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
    # hidden 3
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                64]
    layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
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


######### celebA #########
def celebA_conv_encoder(opts, input, output_dim, reuse=False,
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

def  celebA_conv_decoder(opts, input, output_dim, reuse,
                                            is_training):
    """
    Archi used by Ghosh & al.
    """
    # batch_size
    batch_size = tf.shape(input)[0]
    w,h = output_dim[0], output_dim[1]
    layer_x = input
    # Linear layers
    _out_shape = [int(w/2**4), int(h/2**4), 512]
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                np.prod(_out_shape), scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    layer_x = tf.reshape(layer_x, [-1,] + _out_shape)
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
    # hidden 3
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                64]
    layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid3/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers( opts, layer_x,
                                'hid3/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 4
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                32]
    layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid4/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers( opts, layer_x,
                                'hid4/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=[batch_size,]+output_dim, filter_size=1,
                                stride=1, scope='hid_final/deconv',
                                init= opts['conv_init'])

    return outputs





# CelebA restnet encoder-decoder

def celebA_resnet_encoder(opts, input, output_dim, reuse=False,
                                            is_training=False):
    # batch_size
    layer_x = input

    # first conv layer
    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=64, filter_size=3,
                                stride=1, scope='hid0/conv',
                                init=opts['conv_init'])

    # residual block 1
    layer_res1 = tf.identity(layer_x)

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb1/bn1', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=128, filter_size=3,
                                stride=1, scope='rb1/conv1',
                                init=opts['conv_init'])

    layer_x = tf.nn.avg_pool2d(layer_x, 2, 2, 'VALID')

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb1/bn2', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=128, filter_size=3,
                                stride=1, scope='rb1/conv2',
                                init=opts['conv_init'])

    layer_res1 = tf.nn.avg_pool2d(layer_res1, 2, 2, 'VALID')

    layer_res1 = ops.conv2d.Conv2d(opts, layer_res1, layer_res1.get_shape().as_list()[-1],
                                output_dim=128, filter_size=1,
                                stride=1, scope='rb1/convres',
                                init=opts['conv_init'])

    layer_x = layer_x + layer_res1

    # residual block 2
    layer_res2 = tf.identity(layer_x)

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb2/bn1', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=256, filter_size=3,
                                stride=1, scope='rb2/conv1',
                                init=opts['conv_init'])

    layer_x = tf.nn.avg_pool2d(layer_x, 2, 2, 'VALID')

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb2/bn2', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=256, filter_size=3,
                                stride=1, scope='rb2/conv2',
                                init=opts['conv_init'])

    layer_res2 = tf.nn.avg_pool2d(layer_res2, 2, 2, 'VALID')

    layer_res2 = ops.conv2d.Conv2d(opts, layer_res2, layer_res2.get_shape().as_list()[-1],
                                output_dim=256, filter_size=1,
                                stride=1, scope='rb2/convres',
                                init=opts['conv_init'])

    layer_x = layer_x + layer_res2

    # residual block 3
    layer_res3 = tf.identity(layer_x)

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb3/bn1', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=512, filter_size=3,
                                stride=1, scope='rb3/conv1',
                                init=opts['conv_init'])

    layer_x = tf.nn.avg_pool2d(layer_x, 2, 2, 'VALID')

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb3/bn2', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=512, filter_size=3,
                                stride=1, scope='rb3/conv2',
                                init=opts['conv_init'])

    layer_res3 = tf.nn.avg_pool2d(layer_res3, 2, 2, 'VALID')

    layer_res3 = ops.conv2d.Conv2d(opts, layer_res3, layer_res3.get_shape().as_list()[-1],
                                output_dim=512, filter_size=1,
                                stride=1, scope='rb3/convres',
                                init=opts['conv_init'])

    layer_x = layer_x + layer_res3

    # residual block 4
    layer_res4 = tf.identity(layer_x)

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb4/bn1', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=512, filter_size=3,
                                stride=1, scope='rb4/conv1',
                                init=opts['conv_init'])

    layer_x = tf.nn.avg_pool2d(layer_x, 2, 2, 'VALID')

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb4/bn2', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=512, filter_size=3,
                                stride=1, scope='rb4/conv2',
                                init=opts['conv_init'])

    layer_res4 = tf.nn.avg_pool2d(layer_res4, 2, 2, 'VALID')

    layer_res4 = ops.conv2d.Conv2d(opts, layer_res4, layer_res4.get_shape().as_list()[-1],
                                output_dim=512, filter_size=1,
                                stride=1, scope='rb4/convres',
                                init=opts['conv_init'])

    layer_x = layer_x + layer_res4

    # final layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    outputs = ops.linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                output_dim, scope='hid_final')

    return outputs






def celebA_resnet_decoder(opts, input, output_dim, reuse,
                                            is_training):
    # batch_size
    batch_size = tf.shape(input)[0]
    w,h = output_dim[0], output_dim[1]
    layer_x = input

    # Linear layer
    _out_shape = [int(w/2**4), int(h/2**4), 512]
    layer_x = ops.linear.Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                np.prod(_out_shape), scope='hid0/lin')

    layer_x = tf.reshape(layer_x, [-1,] + _out_shape)

    # Residual block 1: chanel size 512 to 512

    layer_res1 = tf.identity(layer_x)

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb1/bn1', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')


    layer_x = tf.keras.layers.UpSampling2D(2)(layer_x)

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=512, filter_size=3,
                                stride=1, scope='rb1/conv1',
                                init=opts['conv_init'])

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb1/bn2', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=512, filter_size=3,
                                stride=1, scope='rb1/conv2',
                                init=opts['conv_init'])

    layer_res1 = tf.keras.layers.UpSampling2D(2)(layer_res1)

    layer_res1 = ops.conv2d.Conv2d(opts, layer_res1, layer_res1.get_shape().as_list()[-1],
                                output_dim=512, filter_size=1,
                                stride=1, scope='rb1/convres',
                                init=opts['conv_init'])

    layer_x = layer_x + layer_res1

    # Residual block 2: chanel size 512 to 256

    layer_res2 = tf.identity(layer_x)

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb2/bn1', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = tf.keras.layers.UpSampling2D(2)(layer_x)

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=256, filter_size=3,
                                stride=1, scope='rb2/conv1',
                                init=opts['conv_init'])

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb2/bn2', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=256, filter_size=3,
                                stride=1, scope='rb2/conv2',
                                init=opts['conv_init'])

    layer_res2 = tf.keras.layers.UpSampling2D(2)(layer_res2)

    layer_res2 = ops.conv2d.Conv2d(opts, layer_res2, layer_res2.get_shape().as_list()[-1],
                                output_dim=256, filter_size=1,
                                stride=1, scope='rb2/convres',
                                init=opts['conv_init'])

    layer_x = layer_x + layer_res2

    # Residual block 3: channel size 256 to 128

    layer_res3 = tf.identity(layer_x)

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb3/bn1', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = tf.keras.layers.UpSampling2D(2)(layer_x)

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=128, filter_size=3,
                                stride=1, scope='rb3/conv1',
                                init=opts['conv_init'])

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb3/bn2', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=128, filter_size=3,
                                stride=1, scope='rb3/conv2',
                                init=opts['conv_init'])

    layer_res3 = tf.keras.layers.UpSampling2D(2)(layer_res3)

    layer_res3 = ops.conv2d.Conv2d(opts, layer_res3, layer_res3.get_shape().as_list()[-1],
                                output_dim=128, filter_size=1,
                                stride=1, scope='rb3/convres',
                                init=opts['conv_init'])

    layer_x = layer_x + layer_res3

    # Residual block 4: chanel size 128 to 64

    layer_res4 = tf.identity(layer_x)

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb4/bn1', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = tf.keras.layers.UpSampling2D(2)(layer_x)

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=64, filter_size=3,
                                stride=1, scope='rb4/conv1',
                                init=opts['conv_init'])

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'rb4/bn2', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=64, filter_size=3,
                                stride=1, scope='rb4/conv2',
                                init=opts['conv_init'])

    layer_res4 = tf.keras.layers.UpSampling2D(2)(layer_res4)

    layer_res4 = ops.conv2d.Conv2d(opts, layer_res4, layer_res4.get_shape().as_list()[-1],
                                output_dim=64, filter_size=1,
                                stride=1, scope='rb4/convres',
                                init=opts['conv_init'])

    layer_x = layer_x + layer_res4

    # final layers

    if opts['normalization']=='batchnorm':
        layer_x = ops.batchnorm.Batchnorm_layers(opts, layer_x,
                                'fin/bn', is_training, reuse)

    layer_x = ops._ops.non_linear(layer_x,'relu')

    output = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=output_dim[-1], filter_size=3,
                                stride=1, scope='convf',
                                init=opts['conv_init'])

    return output





net_archi = {'mlp': {'encoder': mlp_encoder, 'decoder': mlp_decoder},
            'mnist':{'encoder': mnist_conv_encoder, 'decoder': mnist_conv_decoder},
            'svhn':{'encoder': cifar10_conv_encoder, 'decoder': cifar10_conv_decoder},
            'cifar10':{'encoder': cifar10_conv_encoder, 'decoder': cifar10_conv_decoder},
            'celebA':{'encoder': celebA_resnet_encoder, 'decoder': celebA_resnet_decoder},
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
def singleconv_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # conv
        outputs = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=4,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs

def conv_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=4,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs

def conv_v2_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=3,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=3,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=3,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs

def conv_v3_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=1,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs

def conv_v4_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=64, filter_size=4,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 2
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=256, filter_size=4,
                                    stride=1, scope='hid2/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 3
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=512, filter_size=4,
                                    stride=1, scope='hid3/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=4,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs

def conv_v5_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=3,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=256, filter_size=3,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 2
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=512, filter_size=3,
                                    stride=1, scope='hid2/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=1,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs


######### convdeconv #########
def convdeconv_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    batch_size, in_shape =  tf.shape(inputs)[0], inputs.get_shape().as_list()[1:]
    layer_x = inputs
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=2, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=[batch_size,in_shape[0],in_shape[1],128],
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

def convdeconv_v2_critic(opts, inputs, scope=None, is_training=False, reuse=False):
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

def convdeconv_v3_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    batch_size, in_shape =  tf.shape(inputs)[0], inputs.get_shape().as_list()[1:]
    layer_x = inputs
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=32, filter_size=3,
                                    stride=2, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=[batch_size,in_shape[0],in_shape[1],64],
                                    filter_size=3,
                                    stride=2, scope='hid1/deconv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = ops.deconv2d.Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=[batch_size,]+in_shape, filter_size=1,
                                    stride=1, scope='hid_final/deconv',
                                    init= opts['conv_init'])

    return outputs

def resnet_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # # skip connection
        # layer_x += inputs
        # layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=1,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs + inputs

def resnet_v2_critic(opts, inputs, scope=None, is_training=False, reuse=False):
    layer_x = inputs
    in_shape = inputs.get_shape().as_list()[1:]
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # hidden 0
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid0/conv',
                                    init=opts['conv_init'])
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # hidden 1
        layer_x = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='hid1/conv',
                                    init=opts['conv_init'])
        # skip connection
        shortcut = ops.conv2d.Conv2d(opts, inputs, inputs.get_shape().as_list()[-1],
                                    output_dim=128, filter_size=4,
                                    stride=1, scope='skip/conv',
                                    init=opts['conv_init'])
        layer_x += shortcut
        layer_x = ops._ops.non_linear(layer_x,'leaky_relu')
        # final layer
        outputs = ops.conv2d.Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_dim=in_shape[-1], filter_size=1,
                                    stride=1, scope='hid_final',
                                    init=opts['conv_init'])

    return outputs + inputs


critic_archi = {'mlp': mlp_critic,
            'singleconv': singleconv_critic,
            'conv': conv_critic,
            'conv_v2': conv_v2_critic,
            'conv_v3': conv_v3_critic,
            'conv_v4': conv_v4_critic,
            'conv_v5': conv_v5_critic,
            'convdeconv': convdeconv_critic,
            'convdeconv_v2': convdeconv_v2_critic,
            'convdeconv_v3': convdeconv_v3_critic,
            'resnet': resnet_critic,
            'resnet_v2': resnet_v2_critic
            }
