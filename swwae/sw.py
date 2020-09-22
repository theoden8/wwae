import tensorflow as tf
import tensorflow_probability as tfp
from math import pi


def sw2(opts, x1, x2):
    """
    actually sw1 to test
    """

    h, w, c = x1.get_shape().as_list()[1:]

    sorted_proj_1, x_sorted_1 = distrib_proj(x1, opts['sw_proj_num'], opts['sw_proj_type'])
    sorted_proj_2, x_sorted_2 = distrib_proj(x2, opts['sw_proj_num'], opts['sw_proj_type'])

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

    diff_m = ((mass1 - mass2) / 255.)**2
    diff_m = tf.math.reduce_mean(diff_m, axis=-1)

    return sw + diff_m


def distrib_proj(x, L, law):
    """
    Gets the projected distribution
    """
    h, w, c = x.get_shape().as_list()[1:]
    B = tf.cast(tf.shape(x)[0], tf.int32)
    # get pixel grid projection
    proj = projection(x,L, law) # (L, h*w)
    # sort proj.
    sorted_proj = tf.sort(proj,axis=-1) # (L, h*w)
    # get proj. argsort
    sorted_indices = tf.argsort(proj,axis=-1, stable=True) # (L, h*w)
    # create sort indices
    b_indices = tf.tile(tf.expand_dims(sorted_indices,axis=0),[B,1,1]) # (B,L,h*w)
    bc_indices = tf.tile(tf.expand_dims(b_indices,axis=2),[1,1,c,1]) # (B,L,c,h*w)

    i_b = tf.tile(tf.reshape(tf.range(B), [B,1,1,1]), [1,L,c,h*w])
    i_L = tf.tile(tf.reshape(tf.range(L), [1,L,1,1]), [B,1,c,h*w])
    i_c = tf.tile(tf.reshape(tf.range(c), [1,1,c,1]), [B,L,1,h*w])

    indices = tf.stack([i_b,i_L,i_c,bc_indices], axis=-1)

    # sort im. intensities
    x_flat = tf.transpose(tf.tile(tf.reshape(x, [-1,1,h*w,c]),[1,L,1,1]),[0,1,3,2]) # (batch,L,c,h*w)
    x_sorted = tf.gather_nd(x_flat, indices) #(batch,L,c,h*w)

    return sorted_proj, x_sorted

def projection(x,L,law):
    """
    Wraper to project images pixels gird into the L diferent directions
    return projections coordinates
    """
    # get coor grid
    h, w, c = x.get_shape().as_list()[1:]
    X,Y = tf.meshgrid(tf.range(h), tf.range(w))
    coord = tf.cast(tf.reshape(tf.stack([X,Y],axis=-1),[-1,2]),tf.float32) # ((h*w),2)
    # get directions to project
    if law == 'det':
        thetas = tf.range(L, dtype=tf.float32) / L *pi
    elif law == 'uniform':
        distrib = tfp.distributions.Uniform(low=0., high=pi)
        thetas = distrib.sample(L)
    elif law == 'unidet':
        thetas = tf.range(L, dtype=tf.float32) / L *pi
        distrib = tfp.distributions.Uniform(low=0., high=pi/L)
        shift = distrib.sample(1)
        thetas = thetas + shift
    proj_mat = tf.stack([tf.math.cos(thetas),tf.math.sin(thetas)], axis=-1)
    # project grid into proj dir
    proj = tf.linalg.matmul(proj_mat, coord, transpose_b=True) # (L, (h*w))

    return proj
