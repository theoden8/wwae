import os
from datetime import datetime
import logging
import argparse
import configs
from train import Run
from datahandler import DataHandler
import utils
import itertools

# import tensorflow as tf
import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_eager_execution()
tf.disable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import pdb

parser = argparse.ArgumentParser()
# Args for experiment
parser.add_argument("--model", default='WAE',
                    help='model to train [WAE/BetaVAE/...]')
parser.add_argument('--zdim', type=int, default=2,
                    help='latent dimension')
parser.add_argument("--decoder", default='det',
                    help='decoder typ [det/gauss]')
parser.add_argument("--mode", default='plot',
                    help='mode to run [test/plot]')
parser.add_argument("--dataset", default='mnist',
                    help='dataset')
parser.add_argument("--data_dir", type=str,
                    help='directory in which data is stored')
parser.add_argument("--out_dir", type=str, default='code_outputs',
                    help='root_directory in which outputs are saved')
parser.add_argument("--res_dir", type=str,
                    help='directory in which exp. res are saved')
parser.add_argument("--net_archi", default='conv',
                    help='networks architecture [mlp/conv]')
parser.add_argument("--batchnorm", default='batchnorm',
                    help='batchnormalization')
parser.add_argument("--batch_size", type=int,
                    help='batch size')
parser.add_argument("--beta", type=float, default=0.,
                    help='beta')
parser.add_argument("--gamma", type=float, default=1.,
                    help='weight for mass reg. in ground cost')
parser.add_argument("--cost", default='l2sq',
                    help='ground cost')
parser.add_argument("--weights_file")
## wgan cost
parser.add_argument("--disc_freq", type=int, default=1,
                    help='discriminator update frequency for aversarial sw')
parser.add_argument("--disc_it", type=int, default=10,
                    help='it. num. when updating discriminator for aversarial sw')
parser.add_argument("--critic_archi", type=str, default='fullconv',
                    help='archi for the critic')
parser.add_argument("--critic_pen", type=float, default=1.,
                    help='regularization weight for the critic')
parser.add_argument("--critic_pretrain", action='store_true', default=False,
                    help='pretrain cirtic')
## wemd cost
parser.add_argument("--orientation_num", type=int, default=8,
                    help='oritenations number for waves base')
## sw cost
parser.add_argument("--slicing_dist", type=str, default='det',
                    help='slicing distribution')
parser.add_argument("--L", type=int, default=32,
                    help='Number of slices')

FLAGS = parser.parse_args()


def main():

    # Select dataset to use
    if FLAGS.dataset == 'gmm':
        opts = configs.config_gmm
    elif FLAGS.dataset == 'mnist':
        opts = configs.config_mnist
    elif FLAGS.dataset == 'shifted_mnist':
        opts = configs.config_mnist
        opts['dataset'] = 'shifted_mnist'
    elif FLAGS.dataset == 'shifted_3pos_mnist':
        opts = configs.config_mnist
        opts['dataset'] = 'shifted_3pos_mnist'
    elif FLAGS.dataset == 'rotated_mnist':
        opts = configs.config_mnist
        opts['dataset'] = 'rotated_mnist'
    elif FLAGS.dataset == 'svhn':
        opts = configs.config_svhn
    elif FLAGS.dataset == 'cifar10':
        opts = configs.config_cifar10
    elif FLAGS.dataset == 'celebA':
        opts = configs.config_celeba
    else:
        assert False, 'Unknown dataset'
    # set data_dir
    if FLAGS.data_dir:
        opts['data_dir'] = FLAGS.data_dir
    else:
        raise Exception('You must provide a data_dir')

    ## Model set up
    opts['model'] = FLAGS.model
    if FLAGS.zdim:
        opts['zdim'] = FLAGS.zdim
    opts['decoder'] = FLAGS.decoder
    opts['net_archi'] = FLAGS.net_archi
    if opts['model'][-3:]=='VAE':
        opts['input_normalize_sym'] = False
    if FLAGS.batch_size:
        opts['batch_size'] = FLAGS.batch_size
    opts['normalization'] = FLAGS.batchnorm
    opts['beta'] = FLAGS.beta

    ## ground cost config
    opts['cost'] = FLAGS.cost
    opts['gamma'] = FLAGS.gamma
    # wgan ground cost
    opts['pretrain_critic'] = FLAGS.critic_pretrain
    opts['d_updt_it'] = FLAGS.disc_it
    opts['d_updt_freq'] = FLAGS.disc_freq
    opts['wgan_critic_archi'] = FLAGS.critic_archi
    opts['lambda'] = FLAGS.critic_pen
    # wemd ground cost
    opts['orientation_num'] = FLAGS.orientation_num
    # sw ground cost
    opts['sw_proj_num'] = FLAGS.L
    opts['sw_proj_type'] = FLAGS.slicing_dist

    # Create directories
    results_dir = 'results'
    opts['out_dir'] = os.path.join(results_dir,FLAGS.out_dir)
    out_subdir = os.path.join(opts['out_dir'], opts['model'])
    out_subsubdir = os.path.join(out_subdir, opts['cost']) # + '_' + str(int((FLAGS.id-1) / len(betas))))
    if FLAGS.res_dir:
        exp_name = FLAGS.res_dir
    else:
        exp_name = 'beta_' + str(opts['beta'])
        if opts['cost']=='sw':
            exp_name += '_' + opts['sw_proj_type'] + '_L' + str(opts['sw_proj_num'])
            if opts['sw_proj_type']=='max-sw' or opts['sw_proj_type']=='max-gsw':
                exp_name += '_dfreq' + str(opts['d_updt_freq']) + '_dit' + str(opts['d_updt_it'])
        elif opts['cost']=='wemd':
            exp_name += '_gamma_' + str(opts['gamma'])
            exp_name += '_L_' + str(opts['orientation_num'])
        elif opts['cost']=='l2sq':
            exp_name += '_lrdecay_' + str(opts['lr_decay'])
    opts['exp_dir'] = os.path.join(out_subsubdir, exp_name)
    if not tf.io.gfile.isdir(opts['exp_dir']):
        utils.create_dir(opts['exp_dir'])
        utils.create_dir(os.path.join(opts['exp_dir'], 'checkpoints'))

    # plot setup
    if FLAGS.mode=='plot':
        opts['plot_num_pics'] = 6*6
        opts['plot_num_cols'] = 6

    # Loading the dataset
    data = DataHandler(opts)
    assert data.train_size >= opts['batch_size'], 'Training set too small'

    #Reset tf graph
    tf.compat.v1.reset_default_graph()

    # Loading the dataset
    data = DataHandler(opts)
    assert data.train_size >= opts['batch_size'], 'Training set too small'

    # inti method
    run = Run(opts, data)

    # Training/testing/vizu
    if FLAGS.mode=="test":
        assert False, 'Test not implemented yet.'
        run.test()
    elif FLAGS.mode=="plot":
        run.plot(FLAGS.weights_file)
    elif FLAGS.mode=="perturbation":
        run.perturbation_test(FLAGS.weights_file)
    else:
        assert False, 'Unknown mode %s' % FLAGS.mode

main()
