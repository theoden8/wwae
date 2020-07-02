from math import pi
import tensorflow as tf

import utils

import pdb

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
    n = utils.get_batch_size(sample_qz)
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


def square_dist_v0(sample_x, sample_y):
    """
    Wrapper to compute square distance
    """
    norms_x = tf.reduce_sum(tf.square(sample_x), axis=-1, keepdims=True)
    norms_y = tf.reduce_sum(tf.square(sample_y), axis=-1, keepdims=True)

    squared_dist = norms_x + tf.transpose(norms_y) \
                    - 2. * tf.matmul(sample_x,sample_y,transpose_b=True)
    return tf.nn.relu(squared_dist)


def square_dist(sample_x, sample_y):
    """
    Wrapper 2 to compute square distance
    """
    x = tf.expand_dims(sample_x,axis=1)
    y = tf.expand_dims(sample_y,axis=0)
    squared_dist = tf.reduce_sum(tf.square(x - y),axis=-1)
    return squared_dist

def square_dist_emd(sample_x, sample_y):
    """
    Wrapper 2 to compute square distance on pixel space
    sample_x [batch,d]
    sample_y [batch,d]
    squared_dist [batch,d,d]
    """
    x = tf.expand_dims(sample_x,axis=2)
    y = tf.expand_dims(sample_y,axis=1)
    squared_dist = tf.square(x - y)
    return squared_dist


def ground_cost(opts, x1, x2):
    """
    Compute the WAE's ground cost
    x1: image data             [batch,im_dim]
    x2: image reconstruction   [batch,im_dim]
    """
    # Flatten last dim input
    x1 = tf.layers.flatten(x1)
    x2 = tf.layers.flatten(x2)
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
    cost = tf.reduce_sum(tf.square(x1 - x2), axis=-1)
    cost = tf.sqrt(1e-10 + cost)
    return tf.reduce_mean(cost)


def l2sq_cost(x1,x2):
    # c(x,y) = sum_i(||x - y||_2^2[:,i])
    cost = tf.reduce_sum(tf.square(x1 - x2), axis=-1)
    return tf.reduce_mean(cost)


def l2sq_norm_cost(x1, x2):
    # c(x,y) = mean_i(||x - y||_2^2[:,i])
    cost = tf.reduce_mean(tf.square(x1 - x2), axis=-1)
    return tf.reduce_mean(cost)


def l1_cost(x1, x2):
    # c(x,y) = ||x - y||_1
    cost = tf.reduce_sum(tf.abs(x1 - x2), axis=-1)
    return tf.reduce_mean(cost)


def emd_v0(opts, x1, x2):
    """
    Compute entropy-regularization of the Wasserstein distance
    with shinkhorn algorithm
    """
    L = opts['sinkhorn_iterations']
    eps = opts['sinkhorn_reg']
    C = square_dist_v2(x1, x2)
    # Kernel
    log_K = - C / eps
    # Initialization
    sinkhorn_it = []
    log_v = - tf.math.reduce_logsumexp(log_K, axis=1, keepdims=True)
    # Sinkhorn iterations
    for l in range(L-1):
        log_u = - tf.math.reduce_logsumexp(log_K + log_v, axis=0, keepdims=True)
        sinkhorn_it.append(tf.reduce_sum(tf.exp(log_u+log_K+log_v) * C))
        log_v = - tf.math.reduce_logsumexp(log_K + log_u, axis=1, keepdims=True)
    log_u = - tf.math.reduce_logsumexp(log_K + log_v, axis=0, keepdims=True)
    sinkhorn = tf.reduce_sum(tf.exp(log_u+log_K+log_v) * C)
    sinkhorn_it.append(sinkhorn)
    return sinkhorn, sinkhorn_it

def emd(opts, x1, x2):
    """
    Compute entropy-regularization of the Wasserstein distance
    with shinkhorn algorithm
    """
    # kernel function
    def M(u,v):
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + tf.expand_dims(u, axis=2) + tf.expand_dims(v, axis=1)) / eps
    # params
    n = opts['batch_size']
    d = x1.get_shape().as_list()[-1]
    mu = tf.ones([n,d]) / d
    nu = tf.ones([n,d]) / d
    L = opts['sinkhorn_iterations']
    eps = opts['sinkhorn_reg']
    # normailing images
    x1 /= tf.reduce_sum(x1, axis=-1, keepdims=True)
    x2 /= tf.reduce_sum(x2, axis=-1, keepdims=True)
    # distance matrix
    C = square_dist_emd(x1, x2)
    # Initialization
    sinkhorn_it = []
    u, v = tf.zeros([n,d]), tf.zeros([n,d])
    # Sinkhorn iterations
    for l in range(L):
        u = eps * (tf.log(mu) - tf.squeeze(tf.math.reduce_logsumexp(M(u,v), axis=2))) + u
        v = eps * (tf.log(nu) - tf.squeeze(tf.math.reduce_logsumexp(M(u,v), axis=1))) + v
        sinkhorn_it.append(tf.reduce_mean(tf.reduce_sum(tf.exp(M(u,v)) * C, axis=-1)))
    sinkhorn = tf.reduce_mean(tf.reduce_sum(tf.exp(M(u,v)) * C, axis=-1))
    sinkhorn_it.append(sinkhorn)
    return sinkhorn, sinkhorn_it
