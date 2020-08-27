from math import pi
import tensorflow as tf
import numpy as np
import math as m

from utils import get_batch_size
from ops._ops import logsumexp
from tfp.distributions import Categorical


import pdb

######### latent losses #########
def kl_penalty(pz_mean, pz_sigma, encoded_mean, encoded_sigma):
    """
    Compute KL divergence between prior and variational distribution
    """
    kl = encoded_sigma / pz_sigma \
        + tf.square(pz_mean - encoded_mean) / pz_sigma - 1. \
        + tf.log(pz_sigma) - tf.log(encoded_sigma)
    kl = 0.5 * tf.reduce_sum(kl,axis=-1)
    return tf.reduce_mean(kl)

def cross_entropy_loss(opts, inputs, dec_mean, dec_Sigma):
    if opts['decoder']=='bernoulli':
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=dec_mean)
        cross_entropy = tf.reduce_sum(cross_entropy,axis=-1)
    elif opts['decoder']=='gaussian':
        cross_entropy = tf.log(2*pi) + tf.log(dec_Sigma) + tf.square(inputs-dec_mean) / dec_Sigma
        cross_entropy = -0.5 * tf.reduce_sum(cross_entropy,axis=-1)
    else:
        assert False, 'Cross entropy not implemented for {} decoder' % opts['decoder']
    return tf.reduce_mean(cross_entropy)

def mmd_penalty(opts, sample_qz, sample_pz):
    """
    Comput MMD latent penalty
    """
    sigma2_p = opts['pz_scale'] ** 2
    kernel = opts['mmd_kernel']
    n = get_batch_size(sample_qz)
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)
    half_size = tf.cast((n * n - n) / 2, tf.int32)
    distances_pz = square_dist(sample_pz, sample_pz)
    distances_qz = square_dist(sample_qz, sample_qz)
    distances = square_dist(sample_qz, sample_pz)

    if opts['mmd_kernel'] == 'RBF':
        # Median heuristic for the sigma^2 of Gaussian kernel
        sigma2_k = tf.nn.top_k(
            tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        sigma2_k += tf.nn.top_k(
            tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        # Maximal heuristic for the sigma^2 of Gaussian kernel
        # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
        # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
        # sigma2_k = opts['latent_space_dim'] * sigma2_p
        res1 = tf.exp( - distances_qz / 2. / sigma2_k)
        res1 += tf.exp( - distances_pz / 2. / sigma2_k)
        res1 = tf.multiply(res1, 1. - tf.eye(n))
        res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        res2 = tf.exp( - distances / 2. / sigma2_k)
        res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        stat = res1 - res2
    elif opts['mmd_kernel'] == 'IMQ':
        Cbase = 2 * opts['zdim'] * sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
    elif opts['mmd_kernel'] == 'RQ':
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            res1 = (1. + distances_qz / scale / 2.) ** (-scale)
            res1 += (1. + distances_pz / scale / 2.) ** (-scale)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = (1. + distances / scale / 2.) ** (-scale)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2

    return stat

def square_dist(sample_x, sample_y):
    """
    Wrapper 2 to compute square distance
    """
    x = tf.expand_dims(sample_x,axis=1)
    y = tf.expand_dims(sample_y,axis=0)
    squared_dist = tf.reduce_sum(tf.square(x - y),axis=-1)
    return squared_dist


######### rec losses #########
def ground_cost(opts, x1, x2):
    """
    Compute the WAE's ground cost
    x1: image data             [batch,im_dim]
    x2: image reconstruction   [batch,im_dim]
    """
    # # Flatten last dim input
    # x1 = tf.layers.flatten(x1)
    # x2 = tf.layers.flatten(x2)
    # Compute chosen cost
    if opts['cost'] == 'l2':
        cost = l2_cost(x1, x2)
    elif opts['cost'] == 'l2sq':
        cost = l2sq_cost(x1, x2)
    elif opts['cost'] == 'l2sq_norm':
        cost = l2sq_norm_cost(x1, x2)
    elif opts['cost'] == 'l1':
        cost = l1_cost(x1, x2)
    elif opts['cost'] == 'emd':
        cost, _ = emd(opts, x1, x2)
    else:
        assert False, 'Unknown cost function %s' % opts['cost']
    return cost

def l2_cost(x1, x2):
    # c(x,y) = ||x - y||_2
    cost = tf.reduce_sum(tf.square(x1 - x2), axis=[-3,-2,-1])
    cost = tf.sqrt(1e-10 + cost)
    return tf.reduce_mean(cost)

def l2sq_cost(x1,x2):
    # c(x,y) = sum_i(||x - y||_2^2[:,i])
    cost = tf.reduce_sum(tf.square(x1 - x2), axis=[-3,-2,-1])
    return tf.reduce_mean(cost)

def l2sq_norm_cost(x1, x2):
    # c(x,y) = mean_i(||x - y||_2^2[:,i])
    cost = tf.reduce_mean(tf.square(x1 - x2), axis=[-3,-2,-1])
    return tf.reduce_mean(cost)

def l1_cost(x1, x2):
    # c(x,y) = ||x - y||_1
    cost = tf.reduce_sum(tf.abs(x1 - x2), axis=[-3,-2,-1])
    return tf.reduce_mean(cost)

def emd(opts, x1, x2):
    """
    Compute entropy-regularization of the Wasserstein distance
    with shinkhorn algorithm
    """
    # params
    n = opts['batch_size']
    shape = x1.get_shape().as_list()[1:]
    mu = tf.ones([shape[0]*shape[1],shape[-1]]) / np.prod(shape[:-1])
    nu = tf.ones([shape[0]*shape[1],shape[-1]]) / np.prod(shape[:-1])
    L = opts['sinkhorn_iterations']
    eps = opts['sinkhorn_reg']
    # normailing images
    x1 /= tf.reduce_sum(x1, axis=[1,2], keepdims=True)
    x1 = tf.reshape(x1, [-1,shape[0]*shape[1], shape[-1]])
    x2 /= tf.reduce_sum(x2, axis=[1,2], keepdims=True)
    x2 = tf.reshape(x2, [-1,shape[0]*shape[1], shape[-1]])
    # distance matrix
    C = square_dist_emd(x1, x2)
    # kernel function
    def M(a,b):
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        a = tf.expand_dims(a, axis=2)
        b = tf.expand_dims(b, axis=1)
        return (-C + a + b) / eps
    # Initialization
    sinkhorn_it = []
    u, v = tf.zeros([n,shape[0]*shape[1],shape[-1]]), tf.zeros([n,shape[0]*shape[1],shape[-1]])
    # Sinkhorn iterations
    for l in range(L):
        u = eps * (tf.log(mu) - logsumexp(M(u,v), axis=2, keepdims=False)) + u
        v = eps * (tf.log(nu) - logsumexp(M(u,v), axis=1, keepdims=False)) + v
        sinkhorn_it.append(tf.reduce_mean(tf.reduce_sum(tf.exp(M(u,v)) * C, axis=[-3,-2,-1])))
    sinkhorn = tf.reduce_mean(tf.reduce_sum(tf.exp(M(u,v)) * C, axis=[-3,-2,-1]))
    sinkhorn_it.append(sinkhorn)
    return sinkhorn, sinkhorn_it

def square_dist_emd(x1, x2):
    """
    Wrapper 2 to compute square distance on pixel idx
    x1,2:[batch,w,h,c]: input images
    squared_dist[batch,w,h,w,h,c]: squared_dist[:,i,j,k,l,:] = x1[:,i,j,:]-x2[:,k,l,:]^2
    """
    squared_dist = tf.expand_dims(x1,axis=2)-tf.expand_dims(x2,axis=1)
    return tf.square(squared_dist)

def SW(opts, x1, x2):
    """
    Compute the sliced-wasserstein distance of x1 and x2
    in the pixel space
    x1,2: [batch_size, height, width, channels]
    """
    h, w, c = x.get_shape().as_list()[1:]
    # Get inverse cdf. T are time jumps while w are value of jumps
    T1, w1 = inverse_cdf(opts, x1)
    T2, w2 = inverse_cdf(opts, x2)
    assert w1==w2, 'Error in SW projection'
    # concat and sort time jumps
    T = tf.concat([T1,T2], axis=-2) #(b,L,2*h*w,c)
    idx = tf.argsort(T,axis=-2) #(b,L,2*h*w,c)
    T = tf.sort(T,axis=-2) #(b,L,2*h*w,c)
    # concat, flip sign, and sort jump
    w = tf.concat([w1,-w2], axis=-2) #(L,2*h*w)
    w = tf.reshape
    w = tf.repeat(tf.expand_dims(w,axis=[0,-1]),c,axis=-1) #(L,2*h*w,c)
    w =
    idx_w =
    w = tf.gather_nd

    # TODO

    return sw

def inverse_cdf(opts, x):
    """
    Wraper to compute the inverse cdf distribution on pixel space
    return the ordered jumps positions and jumps values

    cumsum(batch, L, h*w, c): cumsum of the ordered intensities
    sorted_proj(L, h*w): ordered proj. of pixels pos. on L different proj. dir.
    """
    h, w, c = x.get_shape().as_list()[1:]
    batch_size = tf.cast(tf.shape(x)[0], tf.int32)
    L = opts['sw_proj_num']
    # get pixel grid projection
    proj = projection(x, L) # (L, h*w)
    # sort proj.
    sorted_proj = tf.sort(proj,axis=-1) # (L, h*w)
    # get proj. argsort
    sorted_indices = tf.argsort(proj,axis=-1) # (L, h*w)
    # create sorted mask
    range = tf.repeat(tf.expand_dims(tf.range(L),axis=-1), N, axis=-1) #(L,N)
    indices = tf.stack([range,sorted_indices], axis=-1) #(L,N,2)
    batch_indices = tf.repeat(tf.expand_dims(indices,axis=0),batch_size,axis=0)
    # sort im. intensities
    x_flat = tf.reshape(x, [-1,1,h*w,c]) # (batch,1,h*w,c)
    x_sorted = tf.gather_nd(tf.repeat(x_flat,L,axis=1), batch_indices, batch_dims=1) #(batch,L,h*w,c)
    cumsum = tf.math.cumsum(x_sorted,axis=2)

    return cumsum, sorted_proj


def sw2(opts, x1, x2):
    """
    Compute the sliced-wasserstein distance of x1 and x2
    in the pixel space
    x1,2: [batch_size, height, width, channels]
    """
    h, w, c = x.get_shape().as_list()[1:]
    N = opts['sw_samples_num']
    # get distributions approx.
    pc1 = distrib_approx(x1, N)
    pc2 = distrib_approx(x2, N)
    # sort the point clouds
    pc1_sorted = tf.sort(pc1, axis=-1)  # (batch,L,c,N)
    pc2_sorted = tf.sort(pc2, axis=-1)  # (batch,L,c,N)

    sq_diff = tf.math.reduce_mean((pc1_sorted-pc2_sorted)**2, axis=-1)  # (batch,L,c)
    sq_diff = tf.math.reduce_mean(sq_diff, axis=1)  # (batch,c)

    return sq_diff



def distrib_approx(x, N):
    """
    Wraper to approximate the distribution by a sum od Diracs
    """
    h, w, c = x.get_shape().as_list()[1:]
    B = tf.cast(tf.shape(x)[0], tf.int32)
    L = opts['sw_proj_num']
    # projected image
    sorted_proj, x_sorted = distrib_proj(x)  # (L, h*w), (batch,L,c,h*w)
    # expand sorted_proj for batch and channels
    sorted_proj = tf.reshape(sorted_proj,[1,L,1,-1])
    sorted_proj = tf.tile(sorted_proj, [B,1,c,1]) #(batch,L,c,h*w)
    # create the distribution
    dist = tfp.distributions.Categorical(probs=x_sorted)
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


def distrib_proj(x):
    """
    Gets the projected distribution
    """
    h, w, c = x.get_shape().as_list()[1:]
    B = tf.cast(tf.shape(x)[0], tf.int32)
    L = opts['sw_proj_num']
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
    thetas = tf.range(L, dtype=tf.float32) / L *2*np.pi # add other proj methods
    proj_mat = tf.stack([tf.math.cos(thetas),tf.math.sin(thetas)], axis=-1)
    # project grid into proj dir
    proj = tf.linalg.matmul(proj_mat, coord, transpose_b=True) # (L, (h*w))

    return proj
