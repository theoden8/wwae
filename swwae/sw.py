from math import pi
import tensorflow as tf
# import tensorflow_probability as tfp
# import tensorflow_io as tfio

from networks import theta_discriminator

def sw(opts, x1, x2, reuse=False):
    """
    actually sw1 to test
    """

    h, w, c = x1.get_shape().as_list()[1:]

    sorted_proj_1, x_sorted_1 = distrib_proj(opts, x1, reuse=reuse)
    sorted_proj_2, x_sorted_2 = distrib_proj(opts, x2, reuse=True)

    mass1 = tf.reduce_sum(x1, axis=[1,2]) #[b,c]
    xs1 = x_sorted_1 / tf.reshape(mass1, [-1,1,c,1])
    mass2 = tf.reduce_sum(x2, axis=[1,2]) #[b,c]
    xs2 = x_sorted_2 / tf.reshape(mass2, [-1,1,c,1])

    xd_1 = tf.cast(tf.math.cumsum(xs1, axis=-1), tf.float32)
    xd_2 = tf.cast(tf.math.cumsum(xs2, axis=-1), tf.float32)

    z = sorted_proj_1[...,:1]
    z_d = tf.concat((z, sorted_proj_1[...,:-1]), axis=-1)
    steps = sorted_proj_1 - z_d
    steps = tf.expand_dims(steps, axis=0)
    steps = tf.expand_dims(steps, axis=2)


    diff = tf.math.abs(xd_1 - xd_2)*steps #[b,L,c,h*w]
    sw = tf.math.reduce_sum(diff, axis=-1)
    sw = tf.math.reduce_mean(sw, axis=[-2,-1])

    diff_m = (1. - mass2/mass1)**2
    diff_m = tf.math.reduce_mean(diff_m, axis=-1)

    return sw + opts['gamma']*diff_m

def distrib_proj(opts, x, reuse=False):
    """
    Gets the projected distribution
    """
    h, w, c = x.get_shape().as_list()[1:]
    B = tf.cast(tf.shape(x)[0], tf.int32)
    L = opts['sw_proj_num']
    # get pixel grid projection
    proj = projection(opts, x, reuse) # (B,L,c,h*w)
    # sort proj.
    sorted_proj = tf.sort(proj, axis=-1) # (B,L,c, h*w)
    # get proj. argsort
    sorted_indices = tf.argsort(proj, axis=-1, stable=True) # (B,L,c,h*w)
    # create sort indices
#    b_indices = tf.tile(tf.expand_dims(sorted_indices,axis=0),[B,1,1]) # (B,L,h*w)
#    bc_indices = tf.tile(tf.expand_dims(b_indices,axis=2),[1,1,c,1]) # (B,L,c,h*w)

    i_b = tf.tile(tf.reshape(tf.range(B), [B,1,1,1]), [1,L,c,h*w])
    i_L = tf.tile(tf.reshape(tf.range(L), [1,L,1,1]), [B,1,c,h*w])
    i_c = tf.tile(tf.reshape(tf.range(c), [1,1,c,1]), [B,L,1,h*w])

    indices = tf.stack([i_b,i_L,i_c,sorted_indices], axis=-1)

    # sort im. intensities
    x_flat = tf.transpose(tf.tile(tf.reshape(x, [-1,1,h*w,c]), [1,L,1,1]), [0,1,3,2]) # (batch,L,c,h*w)
    x_sorted = tf.gather_nd(x_flat, indices) #(batch,L,c,h*w)

    return sorted_proj, x_sorted

def projection(opts, x, reuse=False):
    """
    Wraper to project images pixels gird into the L diferent directions
    return projections coordinates
    """
    # get coor grid
    h, w, c = x.get_shape().as_list()[1:]
    B = tf.cast(tf.shape(x)[0], tf.int32)
    L = opts['sw_proj_num']
    X,Y = tf.meshgrid(tf.range(h), tf.range(w))
    coord = tf.cast(tf.reshape(tf.stack([X,Y],axis=-1), [-1,2]), tf.float32) # ((h*w),2)
    # get directions to project
    if opts['sw_proj_type']=='det':
        thetas = tf.range(L, dtype=tf.float32) / (pi * L)
        thetas = tf.tile(tf.reshape(thetas, [1,L,1]), [B,1,c])
    elif opts['sw_proj_type']=='uniform':
        # distrib = tfp.distributions.Uniform(low=0., high=pi)
        # thetas = tf.reshape(distrib.sample(B*L*c), [B,L,c])
        thetas = tf.random.uniform([B,L,c], 0., pi)
    elif opts['sw_proj_type']=='unidet':
        thetas = tf.range(L, dtype=tf.float32) / (pi * L)
        thetas = tf.tile(tf.reshape(thetas, [1,L,1]), [B,1,c])
        # distrib = tfp.distributions.Uniform(low=0., high=pi/L)
        # shift = tf.tile(tf.reshape(distrib.sample(B*c), [B,1,c]), [1,L,1])
        shift = tf.random.uniform([B,1,c], 0., pi/L)
        shift = tf.tile(shift, [1,L,1])
        thetas = thetas + shift
    elif opts['sw_proj_type']=='gaussian_small_var':
        thetas = tf.range(L, dtype=tf.float32) / (pi * L)
        thetas = tf.tile(tf.reshape(thetas, [1,L,1]), [B,1,c])
        # distrib = tfp.distributions.Normal(loc=0., scale=pi/L/6)
        # noise = tf.reshape(distrib.sample(B*L*c), [B,L,c])
        noise = tf.random.normal([B,L,c], 0.0, pi/L/6)
        thetas = thetas + noise
    elif opts['sw_proj_type']=='gaussian_large_var':
        thetas = tf.range(L, dtype=tf.float32) / (pi * L)
        thetas = tf.tile(tf.reshape(thetas, [1,L,1]), [B,1,c])
        # distrib = tfp.distributions.Normal(loc=0., scale=pi/L/6)
        # noise = tf.reshape(distrib.sample(B*L*c), [B,L,c])
        noise = tf.random.normal([B,L,c], 0.0, 3*pi/L/6)
        thetas = thetas + noise
    elif opts['sw_proj_type']=='adversarial':
        thetas = theta_discriminator(opts, x, scope='theta_discriminator',
                                    reuse=reuse)
    proj_mat = tf.stack([tf.math.cos(thetas),tf.math.sin(thetas)], axis=-1)
    # project grid into proj dir
    # proj = tf.linalg.matmul(proj_mat, coord, transpose_b=True) # (B,L,c,(h*w))
    proj = tf.compat.v1.matmul(proj_mat, coord, transpose_b=True) # (B,L,c,(h*w))

    return proj
