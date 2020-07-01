import numpy as np
import tensorflow as tf
import math

from networks import encoder, decoder
from datahandler import datashapes
from loss_functions import kl_penalty, cross_entropy_loss, mmd_penalty, ground_cost, emd
import utils

import pdb

class Model(object):

    def __init__(self, opts):
        self.opts = opts

        self.output_dim = datashapes[self.opts['dataset']][:-1] \
                          + [2 * datashapes[self.opts['dataset']][-1], ]

        self.pz_mean = np.zeros(opts['zdim'], dtype='float32')      # TODO don't hardcode this
        self.pz_sigma = np.ones(opts['zdim'], dtype='float32')

    def forward_pass(self, inputs, is_training, dropout_rate, reuse=False):

        enc_z, enc_mean, enc_Sigma = encoder(self.opts,
                                             input=inputs,
                                             output_dim=2 * self.opts['zdim'],
                                             scope='encoder',
                                             reuse=reuse,
                                             is_training=is_training,
                                             dropout_rate=dropout_rate)

        dec_x, dec_mean, dec_Sigma = decoder(self.opts,
                                             input=enc_z,
                                             output_dim=self.output_dim,
                                             scope='decoder',
                                             reuse=reuse,
                                             is_training=is_training,
                                             dropout_rate=dropout_rate)
        return enc_z, enc_mean, enc_Sigma, dec_x, dec_mean, dec_Sigma

    def loss(self, inputs, samples, beta, is_training, dropout_rate):

        # --- Encoding and reconstructing
        enc_z, enc_mean, enc_Sigma, recon_x, dec_mean, dec_Sigma = self.forward_pass(inputs=inputs,
                                                                                is_training=is_training,
                                                                                dropout_rate=dropout_rate)

        loss_reconstruct = self.reconstruction_loss(inputs, recon_x, dec_mean, dec_Sigma)
        match_penalty = beta*self.match_penalty(samples, self.pz_mean, self.pz_sigma, enc_z, enc_mean, enc_Sigma)
        objective = loss_reconstruct + match_penalty

        # --- Pen Encoded Sigma
        if self.opts['pen_enc_sigma'] and self.opts['encoder'] == 'gauss':
            pen_enc_sigma = self.opts['lambda_pen_enc_sigma'] * tf.reduce_mean(
                tf.reduce_sum(tf.abs(tf.log(enc_Sigma)), axis=-1))
            objective += pen_enc_sigma

        # --- Enc Sigma stats
        Sigma_tr = tf.reduce_mean(enc_Sigma, axis=-1)
        Smean, Svar = tf.nn.moments(Sigma_tr, axes=[0])
        encSigmas_stats = tf.stack([Smean, Svar], axis=-1)

        return objective, loss_reconstruct, match_penalty, recon_x, enc_z, encSigmas_stats

    def sample_x_from_prior(self, noise):

        sample_x, sample_mean, sample_Sigma = decoder(self.opts,
                                                      input=noise,
                                                      output_dim=self.output_dim,
                                                      scope='decoder',
                                                      reuse=True,
                                                      is_training=False,
                                                      dropout_rate=1.)
        return sample_x         #, sample_mean, sample_Sigma


class BetaVAE(Model):

    def __init__(self, opts):
        super().__init__(opts)

    def match_penalty(self, pz_samples, pz_mean, pz_sigma, encoded_z, encoded_mean, encoded_sigma): # To check implementation
        """
        Compute KL divergence
        """
        kl = kl_penalty(pz_mean, pz_sigma, encoded_mean, encoded_sigma)
        return kl

    def reconstruction_loss(self, inputs, reconstructions, dec_mean, dec_Sigma):
        """
        Compute Xentropy of decoder
        """
        cross_entropy = cross_entropy_loss(self.opts, inputs, dec_mean, dec_Sigma)
        return cross_entropy


class WAE(Model):

    def __init__(self, opts):
        super().__init__(opts)

    def match_penalty(self, sample_pz, pz_mean, pz_sigma, sample_qz, qz_mean, qz_Sigma):
        """
        Compute MMD latent penalty
        """
        mmd = mmd_penalty(self.opts, sample_qz, sample_pz)
        return mmd

    def reconstruction_loss(self, x1, x2, dec_mean, dec_Sigma):
        """
        Compute ground cost
        """
        opts = self.opts
        # Flatten last dim input
        # - Compute chosen cost
        cost = ground_cost(self.opts, x1, x2)
        return cost

    def sinkhorn(self, x1, x2):
        """
        Compute shinkhorn distance
        """
        x1 = tf.layers.flatten(x1)
        x2 = tf.layers.flatten(x2)
        _, sinkhorn_itertions = emd(self.opts, x1, x2)
        return sinkhorn_itertions
