import numpy as np
import tensorflow as tf
import math

from networks import encoder, decoder
from datahandler import datashapes
from loss_functions import wae_ground_cost, cross_entropy_loss
import utils

import pdb

class Model(object):

    def __init__(self, opts):
        self.opts = opts

        self.output_dim = datashapes[self.opts['dataset']][:-1] \
                          + [2 * datashapes[self.opts['dataset']][-1], ]

        self.pz_mean = np.zeros(opts['zdim'], dtype='float32')      # TODO don't hardcode this
        self.pz_sigma = np.ones(opts['zdim'], dtype='float32')


    def forward_pass(self, inputs, is_training, reuse=False):

        enc_z, enc_mean, enc_Sigma = encoder(self.opts,
                                             input=inputs,
                                             output_dim=2 * self.opts['zdim'],
                                             scope='encoder',
                                             reuse=reuse,
                                             is_training=is_training)

        dec_x, dec_mean, dec_Sigma = decoder(self.opts, input=enc_z,
                                             output_dim=self.output_dim,
                                             scope='decoder',
                                             reuse=reuse,
                                             is_training=is_training)
        return enc_z, enc_mean, enc_Sigma, dec_x, dec_mean, dec_Sigma

    def sample_x_from_prior(self, noise):

        sample_x, _, _ = decoder(self.opts, input=noise, output_dim=self.output_dim,
                                              scope='decoder',
                                              reuse=True,
                                              is_training=False)
        return sample_x

    def MSE(self, inputs, reconstructions):
        # compute MSE between inputs and reconstruction
        square_dist = tf.reduce_sum(tf.square(inputs - reconstructions),axis=[1,2,3])
        return tf.reduce_mean(square_dist)


class BetaVAE(Model):

    def __init__(self, opts):
        super().__init__(opts)

    def kl_penalty(self, pz_mean, pz_sigma, encoded_mean, encoded_sigma): # To check implementation
        """
        Compute KL divergence between prior and variational distribution
        """
        kl = encoded_sigma / pz_sigma \
            + tf.square(pz_mean - encoded_mean) / pz_sigma - 1. \
            + tf.math.log(pz_sigma) - tf.math.log(encoded_sigma)
        kl = 0.5 * tf.reduce_sum(kl, axis=-1)
        return tf.reduce_mean(kl)

    def reconstruction_loss(self, inputs, mean, sigma):
        """
        Compute VAE rec. loss
        """
        rec_loss = cross_entropy_loss(self.opts, inputs, mean, sigma)
        return tf.reduce_mean(rec_loss)

    def loss(self, inputs, beta, is_training):

        _, enc_mean, enc_Sigma, _, dec_mean, dec_Sigma = self.forward_pass(inputs=inputs,
                                              is_training=is_training)

        rec = self.reconstruction_loss(inputs, dec_mean, dec_Sigma)
        kl = self.kl_penalty(self.pz_mean, self.pz_sigma, enc_mean, enc_Sigma)
        reg = beta * kl

        return rec, reg


class WAE(Model):

    def __init__(self, opts):
        super().__init__(opts)

    def square_dist(self, sample_x, sample_y):
        """
        Wrapper to compute square distance
        """
        norms_x = tf.reduce_sum(tf.square(sample_x), axis=-1, keepdims=True)
        norms_y = tf.reduce_sum(tf.square(sample_y), axis=-1, keepdims=True)

        squared_dist = norms_x + tf.transpose(norms_y) \
                        - 2. * tf.matmul(sample_x,sample_y,transpose_b=True)
        return tf.nn.relu(squared_dist)

    def mmd_penalty(self, sample_qz, sample_pz):
        opts = self.opts
        sigma2_p = opts['pz_scale'] ** 2
        kernel = opts['mmd_kernel']
        n = utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = tf.cast((n * n - n) / 2,tf.int32)

        distances_pz = self.square_dist(sample_pz, sample_pz)
        distances_qz = self.square_dist(sample_qz, sample_qz)
        distances = self.square_dist(sample_qz, sample_pz)

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
            if opts['verbose']:
                sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
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

    def reconstruction_loss(self, x1, x2, logits):
        cost = wae_ground_cost(self.opts, x1, x2) #[batch,]

        return tf.reduce_mean(cost)

    def loss(self, inputs, beta, is_training):

        # --- Encoding and reconstructing
        enc_z, _, enc_Sigma, recon_x, dec_mean, _ = self.forward_pass(inputs=inputs,
                                                is_training=is_training)
        rec = self.reconstruction_loss(inputs, recon_x, dec_mean)
        noise = tf.compat.v1.random_normal(shape=tf.shape(enc_z))
        pz_sample = tf.add(self.pz_mean, (noise * self.pz_sigma))
        reg = beta*self.mmd_penalty(enc_z, pz_sample)

        return rec, reg
