
"""
Auto-Encoder models
"""
import os
import sys
import logging

import numpy as np
import tensorflow as tf
from math import ceil

import utils
from sampling_functions import sample_pz, linespace
from plot_functions import save_train, save_test_smallnorb, save_test_celeba, save_dimwise_traversals
from plot_functions import plot_embedded, plot_encSigma, plot_interpolation
import models
from datahandler import datashapes
from fid.fid import calculate_frechet_distance

import pdb

class Run(object):

    def __init__(self, opts, data):

        logging.error('Building the Tensorflow Graph')
        self.opts = opts

        # --- Data
        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        self.data = data

        # --- Initialize prior parameters
        mean = np.zeros(opts['zdim'], dtype='float32')
        Sigma = np.ones(opts['zdim'], dtype='float32')
        self.pz_params = np.concatenate([mean, Sigma], axis=0)

        # --- Placeholders
        self.add_ph()

        # --- Instantiate Model
        if opts['model'] == 'BetaVAE':
            self.model = models.BetaVAE(opts)
        elif opts['model'] == 'WAE':
            self.model = models.WAE(opts)
        else:
            raise NotImplementedError()

        # --- Define Objective
        self.loss_rec, self.loss_reg = self.model.loss(
                                inputs=self.data.next_element,
                                beta=self.beta,
                                is_training=self.is_training)
        self.objective = self.loss_rec + self.loss_reg

        # --- encode & decode pass for testing
        self.z_samples, self.z_mean, self.z_sigma, self.recon_x, _ =\
            self.model.forward_pass(inputs=self.data.next_element,
                                is_training=self.is_training,
                                reuse=True)

        # --- MSE
        self.mse = self.model.MSE(self.data.next_element, self.recon_x)

        # --- Pen Encoded Sigma &  stats
        Sigma_tr = tf.reduce_mean(self.z_sigma, axis=-1)
        Smean, Svar = tf.nn.moments(Sigma_tr, axes=[0])
        self.encSigmas_stats = tf.stack([Smean, Svar], axis=-1)
        if self.opts['pen_enc_sigma'] and self.opts['encoder'] == 'gauss':
            pen_enc_sigma = self.opts['lambda_pen_enc_sigma'] * tf.reduce_mean(
                tf.reduce_sum(tf.abs(tf.math.log(self.z_sigma)), axis=-1))
            self.objective+= pen_enc_sigma

        # --- Get batchnorm ops for training only
        self.extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

        # --- encode & decode pass for vizu
        self.encoded, self.encoded_mean, _, self.decoded, _ =\
            self.model.forward_pass(inputs=self.inputs_img,
                                is_training=self.is_training,
                                reuse=True)

        # --- Sampling
        self.generated = self.model.sample_x_from_prior(noise=self.pz_samples)

        # --- FID score
        if opts['fid']:
            self.inception_graph = tf.Graph()
            self.inception_sess = tf.Session(graph=self.inception_graph)
            with self.inception_graph.as_default():
                self.create_inception_graph()
            self.inception_layer = self._get_inception_layer()

        # --- Optimizers, savers, etc
        self.add_optimizers()

        # --- Init iteratorssess, saver and load trained weights if needed, else init variables
        self.sess = tf.Session()
        self.train_handle, self.test_handle = self.data.init_iterator(self.sess)
        self.saver = tf.train.Saver(max_to_keep=10)
        self.initializer = tf.global_variables_initializer()
        self.sess.graph.finalize()


    def add_ph(self):
        self.lr_decay = tf.compat.v1.placeholder(tf.float32, name='rate_decay_ph')
        self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training_ph')
        self.pz_samples = tf.compat.v1.placeholder(tf.float32,
                                         [None] + [self.opts['zdim'],],
                                         name='noise_ph')
        self.inputs_img = tf.compat.v1.placeholder(tf.float32,
                                         [None] + self.data.data_shape,
                                         name='point_ph')
        self.beta = tf.compat.v1.placeholder(tf.float32, name='beta_ph')

    def compute_blurriness(self):
        images = self.inputs_img
        # First convert to greyscale
        if self.data.data_shape[-1] > 1:
            # We have RGB
            images = tf.image.rgb_to_grayscale(images)
        # Next convolve with the Laplace filter
        lap_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        lap_filter = lap_filter.reshape([3, 3, 1, 1])
        conv = tf.nn.conv2d(images, lap_filter, strides=[1, 1, 1, 1],
                                                padding='VALID')
        _, lapvar = tf.nn.moments(conv, axes=[1, 2, 3])
        return lapvar

    def create_inception_graph(self):
        inception_model = 'classify_image_graph_def.pb'
        inception_path = os.path.join('fid', inception_model)
        # Create inception graph
        with tf.gfile.FastGFile(inception_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString( f.read())
            _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')

    def _get_inception_layer(self):
        # Get inception activation layer (and reshape for batching)
        layername = 'FID_Inception_Net/pool_3:0'
        pool3 = self.inception_sess.graph.get_tensor_by_name(layername)
        ops_pool3 = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops_pool3):
            for o in op.outputs:
                shape = o.get_shape()
                if shape._dims != []:
                  shape = [s.value for s in shape]
                  new_shape = []
                  for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                      new_shape.append(None)
                    else:
                      new_shape.append(s)
                  o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
        return pool3

    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        if opts['optimizer'] == 'sgd':
            return tf.train.GradientDescentOptimizer(lr)
        elif opts['optimizer'] == 'adam':
            return tf.compat.v1.train.AdamOptimizer(lr, beta1=opts['adam_beta1'], beta2=opts['adam_beta2'])
        else:
            assert False, 'Unknown optimizer.'

    def add_optimizers(self):
        opts = self.opts
        lr = opts['lr']
        opt = self.optimizer(lr, self.lr_decay)
        encoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='encoder')
        decoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='decoder')
        with tf.control_dependencies(self.extra_update_ops):
            self.opt = opt.minimize(loss=self.objective, var_list=encoder_vars + decoder_vars)

    def train(self, MODEL_PATH=None, WEIGHTS_FILE=None):
        """
        Train top-down model with chosen method
        """
        logging.error('\nTraining {}'.format(self.opts['model']))
        exp_dir = self.opts['exp_dir']

        # - Set up for training
        train_size = self.data.train_size
        logging.error('\nTrain size: {}, trBatch num.: {}, Ite. num: {}'.format(
                                        train_size,
                                        int(train_size/self.opts['batch_size']),
                                        self.opts['it_num']))
        npics = self.opts['plot_num_pics']
        fixed_noise = sample_pz(self.opts, self.pz_params, npics)
        # anchors_ids = np.random.choice(npics, 5, replace=True)
        anchors_ids = [0, 4, 6, 12, 39]

        # - Init all monitoring variables
        Loss, Loss_test, Loss_rec, Loss_rec_test = [], [], [], []
        Loss_reg, Loss_reg_test = [], []
        MSE, MSE_test = [], []
        if self.opts['vizu_encSigma']:
            enc_Sigmas = []
        # - Init decay lr and lambda
        decay = 1.
        decay_rate = 0.95
        # fix decay
        fix_decay_steps = 500000
        # adaptative decay
        wait = 0
        batches_num = self.data.train_size//self.opts['batch_size']
        ada_decay_steps = batches_num
        # lambda
        wait_lambda = 0

        # - Load trained model or init variables
        if self.opts['use_trained']:
            if MODEL_PATH is None or WEIGHTS_FILE is None:
                    raise Exception("No model/weights provided")
            else:
                if not tf.gfile.IsDirectory(MODEL_PATH):
                    raise Exception("model doesn't exist")
                WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints', WEIGHTS_FILE)
                if not tf.gfile.Exists(WEIGHTS_FILE+".meta"):
                    raise Exception("weights file doesn't exist")
                self.saver.restore(self.sess, WEIGHTS_FILE)
        else:
            self.sess.run(self.initializer)

        # - Training
        for it in range(self.opts['it_num']):
            # Saver
            if it > 0 and it % self.opts['save_every'] == 0:
                self.saver.save(self.sess,
                                os.path.join(exp_dir, 'checkpoints', 'trained-wae'),
                                global_step=it)
            #####  TRAINING LOOP #####
            it += 1
            _ = self.sess.run(self.opt, feed_dict={
                                self.data.handle: self.train_handle,
                                self.lr_decay: decay,
                                self.beta: self.opts['beta'],
                                self.is_training: True})
            ##### TESTING LOOP #####
            if it % self.opts['evaluate_every'] == 0:
                logging.error('\nIteration {}/{}'.format(it, self.opts['it_num']))
                feed_dict={self.data.handle: self.train_handle,
                                self.beta: self.opts['beta'],
                                self.is_training: False}
                [loss, loss_rec, mse, loss_reg] = self.sess.run([
                                            self.objective,
                                            self.loss_rec,
                                            self.mse,
                                            self.loss_reg],
                                            feed_dict=feed_dict)
                Loss.append(loss)
                Loss_rec.append(loss_rec)
                MSE.append(mse)
                Loss_reg.append(loss_reg)

                # Encoded Sigma
                if self.opts['vizu_encSigma']:
                    enc_sigmastats = self.sess.run(self.encSigmas_stats,
                                            feed_dict=feed_dict)
                    enc_Sigmas.append(enc_sigmastats)

                test_it_num = int(10000 / self.opts['batch_size'])
                loss, loss_rec, mse, loss_reg = 0., 0., 0., 0.
                for it_ in range(test_it_num):
                    test_feed_dict={self.data.handle: self.test_handle,
                                    self.beta: self.opts['beta'],
                                    self.is_training: False}
                    [l, l_rec, m, l_reg] = self.sess.run([self.objective,
                                                self.loss_rec,
                                                self.mse,
                                                self.loss_reg],
                                                feed_dict=test_feed_dict)
                    loss += l / test_it_num
                    loss_rec += l_rec / test_it_num
                    mse += m / test_it_num
                    loss_reg += l_reg / test_it_num
                Loss_test.append(loss)
                Loss_rec_test.append(loss_rec)
                MSE_test.append(mse)
                Loss_reg_test.append(loss_reg)

                # Printing various loss values
                debug_str = 'ITER: %d/%d, ' % (it, self.opts['it_num'])
                logging.error(debug_str)
                debug_str = 'TRAIN LOSS=%.3f, TEST LOSS=%.3f' % (Loss[-1],Loss_test[-1])
                logging.error(debug_str)
                debug_str = 'REC=%.3f, TEST REC=%.3f, MSE=%10.3e, TEST MSE=%10.3e, \n '  % (
                                            Loss_rec[-1],
                                            Loss_rec_test[-1],
                                            MSE[-1],
                                            MSE_test[-1])
                logging.error(debug_str)
                if self.opts['model'] == 'BetaVAE':
                    debug_str = 'beta*KL=%10.3e, beta*TEST KL=%10.3e, \n '  % (
                                                Loss_reg[-1],
                                                Loss_reg_test[-1])
                    logging.error(debug_str)
                elif self.opts['model'] == 'WAE':
                    debug_str = 'beta*MMD=%10.3e, beta*TEST MMD=%10.3e \n ' % (
                                                Loss_reg[-1],
                                                Loss_reg_test[-1])
                    logging.error(debug_str)
                else:
                    raise NotImplementedError('Model type not recognised')
            ##### Vizu LOOP #####
            if it % self.opts['print_every'] == 0:
                # Plot vizualizations
                # Auto-encoding test images & samples generated by the model
                [reconstructions_vizu, latents_vizu, generations] = self.sess.run(
                                            [self.decoded, self.encoded, self.generated],
                                            feed_dict={self.inputs_img: self.data.data_vizu,
                                                       self.pz_samples: fixed_noise,
                                                       self.is_training: False})
                # - Plotting latent interpolation, and saving

                # Encoded sigma
                if self.opts['vizu_encSigma'] and it > 1:
                    plot_encSigma(self.opts,
                                  enc_Sigmas,
                                  exp_dir,
                                  'encSigma_e%04d_mb%05d.png' % (epoch, buff*batches_num+it+1))

                # Encode anchors points and interpolate
                if self.opts['vizu_interpolation']:
                    num_steps = 15

                    enc_var = np.ones(self.opts['zdim'])
                    # crate linespace
                    enc_interpolation = linespace(self.opts, num_steps,  # shape: [nanchors, zdim, nsteps, zdim]
                                            anchors=latents_vizu[anchors_ids],
                                            std=enc_var)
                    enc_interpolation = np.reshape(enc_interpolation, [-1, self.opts['zdim']])
                    # reconstructing
                    dec_interpolation = self.sess.run(self.generated,
                                            feed_dict={self.pz_samples: enc_interpolation,
                                                       self.is_training: False})
                    inter_anchors = np.reshape(dec_interpolation, [-1, self.opts['zdim'], num_steps]+self.data.data_shape)
                    kl_to_prior_sorted = np.argsort(kl)[::-1]
                    plot_interpolation(self.opts, inter_anchors[:,kl_to_prior_sorted],
                                            exp_dir, 'inter_it%07d.png' % (it))

                # Auto-encoding training images
                inputs_tr = []
                n_next_element = ceil(npics / self.opts['batch_size'])
                for _ in range(n_next_element):
                    inputs_tr.append(self.sess.run(self.data.next_element, feed_dict={self.data.handle: self.train_handle}))
                inputs_tr = np.concatenate(inputs_tr,axis=0)[:npics]
                reconstructions_train = self.sess.run(self.decoded,
                                            feed_dict={self.inputs_img: inputs_tr,
                                                       self.is_training: False})


                # Saving plots
                save_train(self.opts,
                          inputs_tr, self.data.data_vizu,                       # train/vizu images
                          reconstructions_train, reconstructions_vizu,          # reconstructions
                          generations,                                          # model samples
                          Loss, Loss_test,                                      # loss
                          Loss_rec, Loss_rec_test,                              # rec loss
                          MSE, MSE_test,                                        # mse
                          Loss_reg, Loss_reg_test,                              # divergence terms
                          exp_dir,                                              # working directory
                          'res_it%07d.png' % (it))                              # filename

            # - Update learning rate if necessary and it
            if self.opts['lr_decay']:
                """
                if (it+1) % fix_decay_steps == 0:
                    decay = decay_rate ** (int(it / fix_decay_steps))
                    logging.error('Reduction in lr: %f\n' % decay)
                """
                    # If no significant progress was made in the last epoch
                    # then decrease the learning rate.
                if np.mean(Loss_rec[-ada_decay_steps:]) < np.mean(Loss_rec[-5*ada_decay_steps:]) - np.var(Loss_rec[-5*ada_decay_steps:]):
                    wait = 0
                else:
                    wait += 1
                if wait > ada_decay_steps:
                    decay = decay_rate ** (int(it / ada_decay_steps))
                    logging.error('Reduction in lr: %f\n' % decay)
                    wait = 0


            # - Update regularizer if necessary
            if self.opts['lambda_schedule'] == 'adaptive':
                if it >= .0 and len(Loss_rec) > 0:
                    if wait_lambda > 1000 * batches_num + 1:
                        # opts['lambda'] = list(2*np.array(opts['lambda']))
                        self.opts['lambda'][-1] = 2*self.opts['lambda'][-1]
                        wae_lambda = self.opts['lambda']
                        logging.error('Lambda updated to %s\n' % wae_lambda)
                        print('')
                        wait_lambda = 0
                    else:
                        wait_lambda += 1

            # - logging
            if (it)%50000 ==0 :
                logging.error('Train it.: {}/{} \n'.format(it,self.opts['it_num']))

        # - Save the final model
        if self.opts['save_final'] and it > 0:
            self.saver.save(self.sess, os.path.join(exp_dir,
                                                'checkpoints',
                                                'trained-{}-final'.format(self.opts['model'])),
                                                global_step=it)

        # - Finale losses & scores
        feed_dict={self.data.handle: self.train_handle,
                        self.beta: self.opts['beta'],
                        self.is_training: False}
        [loss, loss_rec, mse, loss_reg] = self.sess.run([
                                    self.objective,
                                    self.loss_rec,
                                    self.mse,
                                    self.loss_reg],
                                    feed_dict=feed_dict)
        Loss.append(loss)
        Loss_rec.append(loss_rec)
        MSE.append(mse)
        Loss_reg.append(loss_reg)
        # Test losses
        loss, loss_rec, mse, loss_reg= 0., 0., 0., 0.
        for it_ in range(test_it_num):
            test_feed_dict={self.data.handle: self.test_handle,
                            self.beta: self.opts['beta'],
                            self.is_training: False}
            [l, l_rec, m, l_reg] = self.sess.run([self.objective,
                                        self.loss_rec,
                                        self.mse,
                                        self.loss_reg],
                                        feed_dict=test_feed_dict)
            loss += l / test_it_num
            loss_rec += l_rec / test_it_num
            mse += m / test_it_num
            loss_reg += l_reg / test_it_num
        Loss_test.append(loss)
        Loss_rec_test.append(loss_rec)
        MSE_test.append(mse)
        Loss_reg_test.append(loss_reg)

        # Printing various loss values
        logging.error('Training done.')
        debug_str = 'TRAIN LOSS=%.3f, TEST LOSS=%.3f' % (Loss[-1],Loss_test[-1])
        logging.error(debug_str)
        debug_str = 'REC=%.3f, TEST REC=%.3f, MSE=%10.3e, TEST MSE=%10.3e, \n '  % (
                                    Loss_rec[-1],
                                    Loss_rec_test[-1],
                                    MSE[-1],
                                    MSE_test[-1])
        logging.error(debug_str)
        if self.opts['model'] == 'BetaVAE':
            debug_str = 'beta*KL=%10.3e, beta*TEST KL=%10.3e, \n '  % (
                                        Loss_reg[-1],
                                        Loss_reg_test[-1])
            logging.error(debug_str)
        elif self.opts['model'] == 'WAE':
            debug_str = 'beta*MMD=%10.3e, beta*TEST MMD=%10.3e \n ' % (
                                        Loss_reg[-1],
                                        Loss_reg_test[-1])
            logging.error(debug_str)
        else:
            raise NotImplementedError('Model type not recognised')

        # - save training data
        if self.opts['save_train_data']:
            data_dir = 'train_data'
            save_path = os.path.join(exp_dir, data_dir)
            utils.create_dir(save_path)
            name = 'res_train_final'
            np.savez(os.path.join(save_path, name),
                    loss=np.array(Loss[-1]), loss_test=np.array(Loss_test[-1]),
                    loss_rec=np.array(Loss_rec[-1]), loss_rec_test=np.array(Loss_rec_test[-1]),
                    mse = np.array(MSE[-1]), mse_test = np.array(MSE_test[-1]),
                    loss_reg=np.array(Loss_reg[-1]), loss_reg_test=np.array(Loss_reg_test[-1]))

    def test(self, data, WEIGHTS_PATH, verbose):
        """
        Test model and save different metrics
        """

        opts = self.opts

        # - Load trained weights
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)

        # - Set up
        test_size = data.test_size
        batch_size_te = min(test_size,1000)
        batches_num_te = int(test_size/batch_size_te)+1
        # - Init all monitoring variables
        Loss, Loss_rec, MSE = 0., 0., 0.
        Divergences = []
        MIG, factorVAE, SAP = 0., 0., 0.
        real_blurr, blurr, fid_scores = 0., 0., 0.
        if opts['true_gen_model']:
            codes, codes_mean = np.zeros((batches_num_te*batch_size_te,opts['zdim'])), np.zeros((batches_num_te*batch_size_te,opts['zdim']))
            labels = np.zeros((batches_num_te*batch_size_te,len(data.factor_indices)))
        # - Testing loop
        for it_ in range(batches_num_te):
            # Sample batches of data points
            data_ids = np.random.choice(test_size, batch_size_te, replace=True)
            batch_images_test = data.get_batch_img(data_ids, 'test').astype(np.float32)
            batch_pz_samples_test = sample_pz(opts, self.pz_params, batch_size_te)
            test_feed_dict = {self.batch: batch_images_test,
                              self.samples_pz: batch_pz_samples_test,
                              self.obj_fn_coeffs: opts['obj_fn_coeffs'],
                              self.is_training: False}
            [loss, l_rec, mse, divergences, z, z_mean, samples] = self.sess.run([self.objective,
                                             self.loss_reconstruct,
                                             self.mse,
                                             self.divergences,
                                             self.z_samples,
                                             self.z_mean,
                                             self.generated_x],
                                            feed_dict=test_feed_dict)
            Loss += loss / batches_num_te
            Loss_rec += l_rec / batches_num_te
            MSE += mse / batches_num_te
            if len(Divergences)>0:
                Divergences[-1] += np.array(divergences) / batches_num_te
            else:
                Divergences.append(np.array(divergences) / batches_num_te)
            # storing labels and factors
            if opts['true_gen_model']:
                    codes[batch_size_te*it_:batch_size_te*(it_+1)] = z
                    codes_mean[batch_size_te*it_:batch_size_te*(it_+1)] = z_mean
                    labels[batch_size_te*it_:batch_size_te*(it_+1)] = data.get_batch_label(data_ids,'test')[:,data.factor_indices]
            # fid score
            if opts['fid']:
                # Load inception mean samples for train set
                trained_stats = os.path.join(inception_path, 'fid_stats.npz')
                # Load trained stats
                f = np.load(trained_stats)
                self.mu_train, self.sigma_train = f['mu'][:], f['sigma'][:]
                f.close()
                # Compute bluriness of real data
                real_blurriness = self.sess.run(self.blurriness,
                                            feed_dict={ self.batch: batch_images_test})
                real_blurr += np.mean(real_blurriness) / batches_num_te
                # Compute gen blur
                gen_blurr = self.sess.run(self.blurriness,
                                            feed_dict={self.batch: samples})
                blurr += np.mean(gen_blurr) / batches_num_te
                # Compute FID score
                # First convert to RGB
                if np.shape(samples)[-1] == 1:
                    # We have greyscale
                    samples = self.sess.run(tf.image.grayscale_to_rgb(samples))
                preds_incep = self.inception_sess.run(self.inception_layer,
                              feed_dict={'FID_Inception_Net/ExpandDims:0': samples})
                preds_incep = preds_incep.reshape((batch_size_te,-1))
                mu_gen = np.mean(preds_incep, axis=0)
                sigma_gen = np.cov(preds_incep, rowvar=False)
                fid_score = fid.calculate_frechet_distance(mu_gen, sigma_gen,
                                            self.mu_train,
                                            self.sigma_train,
                                            eps=1e-6)
                fid_scores += fid_score / batches_num_te
        # - Compute disentanglment metrics
        if opts['true_gen_model']:
            MIG.append(self.compute_mig(codes_mean, labels))
            factorVAE.append(self.compute_factorVAE(data, codes))
            SAP.append(self.compute_SAP(data))

        # - Printing various loss values
        if verbose=='high':
            debug_str = 'Testing done.'
            logging.error(debug_str)
            if opts['true_gen_model']:
                debug_str = 'MIG=%.3f, factorVAE=%.3f, SAP=%.3f' % (
                                            MIG,
                                            factorVAE,
                                            SAP)
                logging.error(debug_str)
            if opts['fid']:
                debug_str = 'Real blurr=%10.3e, blurr=%10.3e, FID=%.3f \n ' % (
                                            real_blurr,
                                            blurr,
                                            fid_scores)
                logging.error(debug_str)

            if opts['model'] == 'BetaVAE':
                debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, beta*KL=%10.3e \n '  % (
                                            Loss,
                                            Loss_rec,
                                            MSE,
                                            Divergences)
                logging.error(debug_str)
            elif opts['model'] == 'BetaTCVAE':
                debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, b*TC=%10.3e, KL=%10.3e \n '  % (
                                            Loss,
                                            Loss_rec,
                                            MSE,
                                            Divergences[0],
                                            Divergences[1])
                logging.error(debug_str)
            elif opts['model'] == 'FactorVAE':
                debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, b*KL=%10.3e, g*TC=%10.3e, \n '  % (
                                            Loss,
                                            Loss_rec,
                                            MSE,
                                            Divergences[0],
                                            Divergences[1])
                logging.error(debug_str)
            elif opts['model'] == 'WAE':
                debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, b*MMD=%10.3e \n ' % (
                                            Loss,
                                            Loss_rec,
                                            MSE,
                                            Divergences)
                logging.error(debug_str)
            elif opts['model'] == 'disWAE':
                debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, b*HSIC=%10.3e, g*DIMWISE=%10.3e, WAE=%10.3e' % (
                                            Loss,
                                            Loss_rec,
                                            MSE,
                                            Divergences[0],
                                            Divergences[1],
                                            Divergences[2])
                logging.error(debug_str)
            elif opts['model'] == 'TCWAE_MWS' or opts['model'] == 'TCWAE_GAN':
                debug_str = 'LOSS=%.3f, REC=%.3f,l1*TC=%10.3e, MSE=%.3f, l2*DIMWISE=%10.3e, WAE=%10.3e' % (
                                            Loss,
                                            Loss_rec,
                                            MSE,
                                            Divergences[0],
                                            Divergences[1],
                                            Divergences[2])
                logging.error(debug_str)
            else:
                raise NotImplementedError('Model type not recognised')


        # - save testing data
        data_dir = 'test_data'
        save_path = os.path.join(opts['exp_dir'], data_dir)
        utils.create_dir(save_path)
        name = 'res_test_final'
        np.savez(os.path.join(save_path, name),
                loss=np.array(Loss),
                loss_rec=np.array(Loss_rec),
                mse = np.array(MSE),
                divergences=Divergences,
                mig=np.array(MIG),
                factorVAE=np.array(factorVAE),
                sap=np.array(SAP),
                real_blurr=np.array(real_blurr),
                blurr=np.array(blurr),
                fid=np.array(fid_scores))

    def plot(self, data, WEIGHTS_PATH):
        """
        Plots reconstructions, latent transversals and model samples
        """

        opts = self.opts

        # - Load trained weights
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)

        # - Set up
        im_shape = datashapes[opts['dataset']]
        if opts['dataset']=='celebA' or opts['dataset']=='3Dchairs':
            num_pics = 100
            num_steps = 7
        else:
            num_pics = 20
            num_steps = 10
        fixed_noise = sample_pz(opts, self.pz_params, num_pics)


        # - Auto-encoding test images & samples generated by the model
        [reconstructions, latents, generations] = self.sess.run(
                                    [self.recon_x,
                                     self.z_samples,
                                     self.generated_x],
                                    feed_dict={self.batch: data.data[self.plot_data_idx],
                                               self.samples_pz: fixed_noise,
                                               self.is_training: False})
        # - get kl(q(z_i),p(z_i)) on test data to plot latent traversals
        test_size = self.test_size
        batch_size_te = min(test_size,1000)
        batches_num_te = int(test_size / batch_size_te)+1
        kl_to_prior = np.zeros(opts['zdim'])
        for it_ in range(batches_num_te):
            data_ids = np.random.choice(test_size, batch_size_te, replace=True)
            batch_images_test = data.get_batch_img(data_ids, 'test').astype(np.float32)
            kl = self.sess.run(self.kl_to_prior, feed_dict={
                                                self.batch: batch_images_test,
                                                self.is_training: False})
            kl_to_prior += kl / batches_num_te

        # - Latent transversals
        enc_var = np.ones(opts['zdim'])
        # create latent linespacel
        if opts['dataset']=='celebA' :
            idx = [0,3,20,26,40,49]
            latent_transversal = linespace(opts, num_steps,  # shape: [nanchors, zdim, nsteps, zdim]
                                    anchors=latents[idx],
                                    std=enc_var)
            # latent_transversal = latent_transversal[:,:,::-1]
        elif opts['dataset']=='3Dchairs':
            idx = [48,4,21,44,1]
            latent_transversal = linespace(opts, num_steps,  # shape: [nanchors, zdim, nsteps, zdim]
                                    anchors=latents[idx],
                                    std=enc_var)
            latent_transversal = latent_transversal[:,:,::-1]
        else:
            latent_transversal = linespace(opts, num_steps,  # shape: [nanchors, zdim, nsteps, zdim]
                                    anchors=latents[::2],
                                    std=enc_var)
        # - Reconstructing latent transversals
        obs_transversal = self.sess.run(self.generated_x,
                                    feed_dict={self.samples_pz: np.reshape(latent_transversal,[-1, opts['zdim']]),
                                               self.is_training: False})
        obs_transversal = np.reshape(obs_transversal, [-1, opts['zdim'], num_steps]+im_shape)
        kl_to_prior_sorted = np.argsort(kl_to_prior)[::-1]
        obs_transversal = obs_transversal[:,kl_to_prior_sorted]

        # - ploting and saving
        if opts['dataset']=='celebA' or opts['dataset']=='3Dchairs':
            save_test_celeba(opts, data.data[self.plot_data_idx],
                                        reconstructions,
                                        obs_transversal,
                                        generations,
                                        opts['exp_dir'])
            save_dimwise_traversals(opts, obs_transversal, opts['exp_dir'])
        else:
            save_test_smallnorb(opts, data.data[self.plot_data_idx],
                                        reconstructions,
                                        obs_transversal,
                                        generations,
                                        opts['exp_dir'])

    def fid_score(self, MODEL_PATH, WEIGHTS_FILE, COMPUTE_DATASET_STASTICS, fid_inputs='samples'):
        """
        Compute FID score
        """

        opts = self.opts

        # --- Load trained weights
        if not tf.gfile.IsDirectory(MODEL_PATH):
            raise Exception("model doesn't exist")
        WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)

        if self.data.dataset == 'celebA':
            compare_dataset_name = 'celeba_stats.npz'
        elif self.data.dataset == 'cifar10':
            compare_dataset_name = 'cifar10_stats.npz'
        elif self.data.dataset == 'svhn':
            compare_dataset_name = 'svhn_stats.npz'
        elif self.data.dataset == 'mnist':
            compare_dataset_name = 'mnist_stats.npz'
        else:
            assert False, 'FID not implemented for {} dataset.'.format(self.data.dataset)

        # --- setup
        full_size = self.data.train_size + self.data.test_size
        test_size = 1000
        batch_size = 100
        batch_num = int(test_size/batch_size)
        fid_dir = 'fid'
        stats_path = os.path.join(fid_dir, compare_dataset_name)

        # --- Compute stats on real dataset if needed
        if COMPUTE_DATASET_STASTICS:
            preds_list = []
            for n in range(batch_num):
                batch_id = np.random.randint(full_size, size=batch_size)
                batch = self.data._sample_observations(batch_id)
                # Convert to RGB if needed
                if np.shape(batch)[-1] == 1:
                    batch = np.repeat(batch, 3, axis=-1)
                preds_incep = self.inception_sess.run(self.inception_layer,
                              feed_dict={'FID_Inception_Net/ExpandDims:0': batch})
                preds_incep = preds_incep.reshape((batch_size,-1))
                preds_list.append(preds_incep)
            preds_list = np.concatenate(preds_list, axis=0)
            mu = np.mean(preds_list, axis=0)
            sigma = np.cov(preds_list, rowvar=False)
            # saving stats
            np.savez(stats_path, m=mu, s=sigma)
        else:
            if not os.path.isfile(stats_path):
                raise NotImplementedError('No statistics found for given dataset')
            stats = np.load(stats_path)
            mu = stats['m']
            sigma = stats['s']

        # --- Compute stats for reconstructions or samples
        if fid_inputs == 'reconstruction':
            preds_list = []
            for n in range(batch_num):
                batch_id = np.random.randint(full_size, size=batch_size)
                batch = self.data._sample_observations(batch_id)
                recons = self.sess.run(self.decoded, feed_dict={
                                        self.inputs_img: batch,
                                        self.is_training: False})
                if np.shape(recons)[-1] == 1:
                    recons = np.repeat(recons, 3, axis=-1)
                preds_incep = self.inception_sess.run(self.inception_layer,
                              feed_dict={'FID_Inception_Net/ExpandDims:0': recons})
                preds_incep = preds_incep.reshape((batch_size,-1))
                preds_list.append(preds_incep)
            preds_list = np.concatenate(preds_list, axis=0)
            mu_model = np.mean(preds_list, axis=0)
            sigma_model = np.cov(preds_list, rowvar=False)
        elif fid_inputs == 'samples':
            preds_list = []
            for n in range(batch_num):
                batch = sample_pz(opts, self.pz_params, batch_size)
                samples = self.sess.run(self.generated, feed_dict={
                                        self.pz_samples: batch,
                                        self.is_training: False})
                if np.shape(samples)[-1] == 1:
                    samples = np.repeat(samples, 3, axis=-1)
                preds_incep = self.inception_sess.run(self.inception_layer,
                              feed_dict={'FID_Inception_Net/ExpandDims:0': samples})
                preds_incep = preds_incep.reshape((batch_size,-1))
                preds_list.append(preds_incep)
            preds_list = np.concatenate(preds_list, axis=0)
            mu_model = np.mean(preds_list, axis=0)
            sigma_model = np.cov(preds_list, rowvar=False)

        # --- Compute FID between real stats and model stats
        fid_scores = calculate_frechet_distance(mu, sigma, mu_model, sigma_model)

        # --- Logging
        debug_str = 'FID={:.3f} for {} data'.format(fid_scores, test_size)
        logging.error(debug_str)
        fid_res_dir = os.path.join(MODEL_PATH,'fid')
        if not tf.io.gfile.isdir(fid_res_dir):
            utils.create_dir(fid_res_dir)
        filename = 'fid_' + fid_inputs
        np.save(os.path.join(fid_res_dir,filename),fid_scores)
