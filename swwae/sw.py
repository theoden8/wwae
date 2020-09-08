import tensorflow as tf
import tensorflow_probability as tfp
from math import pi

def sw2(opts, x1, x2):
    """
    Compute the sliced-wasserstein distance of x1 and x2
    in the pixel space
    x1,2: [batch_size, height, width, channels]
    """
    N = opts['sw_samples_num']
    # get distributions approx.
    pc1 = distrib_approx(x1, opts['sw_proj_num'], opts['sw_samples_num'])
    pc2 = distrib_approx(x2, opts['sw_proj_num'], opts['sw_samples_num'])
    # sort the point clouds
    pc1_sorted = tf.sort(pc1, axis=-1)  # (batch,L,c,N)
    pc2_sorted = tf.sort(pc2, axis=-1)  # (batch,L,c,N)

    sq_diff = tf.math.reduce_mean((pc1_sorted-pc2_sorted)**2, axis=-1)  # (batch,L,c)
    sq_diff = tf.math.reduce_mean(sq_diff, axis=1)  # (batch,c)
    # we take the mean over the chanels for the moment
    sq_diff = tf.math.reduce_mean(sq_diff, axis=1)  # (batch,)

    return sq_diff

def distrib_approx(x, L, N):
    """
    Wraper to approximate the distribution by a sum of Diracs
    x: inputs [batch_size, height, width, channels]
    L: num of projections
    N: num of samples
    """
    h, w, c = x.get_shape().as_list()[1:]
    B = tf.cast(tf.shape(x)[0], tf.int32)
    # projected image
    sorted_proj, x_sorted = distrib_proj(x, L)  # (L, h*w), (batch,L,c,h*w)
    # expand sorted_proj for batch and channels
    sorted_proj = tf.reshape(sorted_proj,[1,L,1,-1])
    sorted_proj = tf.tile(sorted_proj, [B,1,c,1]) #(batch,L,c,h*w)
    # create the distribution
    mass = tf.reduce_sum(x_sorted, axis=-1, keepdims=True)
    dist = tfp.distributions.Categorical(probs=x_sorted/mass)
    # sample from the distribution N times
    samples = dist.sample(N) # (N,batch,L,c)
    samples = tf.transpose(samples, [1,2,3,0])

    i_b = tf.tile(tf.reshape(tf.range(B), [B,1,1,1]), [1,L,c,N])
    i_L = tf.tile(tf.reshape(tf.range(L), [1,L,1,1]), [B,1,c,N])
    i_c = tf.tile(tf.reshape(tf.range(c), [1,1,c,1]), [B,L,1,N])

    indices = tf.stack([i_b,i_L,i_c,samples], axis=-1)
    #from the samples, get the pixel values
    point_cloud = tf.gather_nd(sorted_proj, indices)  #(batch,L,c,N)

    return point_cloud


def distrib_proj(x, L):
    """
    Gets the projected distribution
    """
    h, w, c = x.get_shape().as_list()[1:]
    B = tf.cast(tf.shape(x)[0], tf.int32)
    # get pixel grid projection
    proj = projection(x,L) # (L, h*w)
    # sort proj.
    sorted_proj = tf.sort(proj,axis=-1) # (L, h*w)
    # get proj. argsort
    sorted_indices = tf.argsort(proj,axis=-1) # (L, h*w)
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


def projection(x,L):
    """
    Wraper to project images pixels gird into the L diferent directions
    return projections coordinates
    """
    # get coor grid
    h, w, c = x.get_shape().as_list()[1:]
    X,Y = tf.meshgrid(tf.range(h), tf.range(w))
    coord = tf.cast(tf.reshape(tf.stack([X,Y],axis=-1),[-1,2]),tf.float32) # ((h*w),2)
    # get directions to project
    thetas = tf.range(L, dtype=tf.float32) / L *2*pi # add other proj methods
    proj_mat = tf.stack([tf.math.cos(thetas),tf.math.sin(thetas)], axis=-1)
    # project grid into proj dir
    proj = tf.linalg.matmul(proj_mat, coord, transpose_b=True) # (L, (h*w))

    return proj
