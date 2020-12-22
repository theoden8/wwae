import os
from datetime import datetime
import logging
import argparse
import configs
from train import Run
from datahandler import DataHandler
import utils
import itertools

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import pdb

parser = argparse.ArgumentParser()
# Args for experiment
parser.add_argument("--model", default='WAE',
                    help='model to train [WAE/BetaVAE/...]')
parser.add_argument("--decoder", default='det',
                    help='decoder typ [det/gauss]')
parser.add_argument("--mode", default='train',
                    help='mode to run [train/vizu/fid/test]')
parser.add_argument("--dataset", default='mnist',
                    help='dataset')
parser.add_argument("--data_dir", type=str,
                    help='directory in which data is stored')
parser.add_argument("--out_dir", type=str, default='code_outputs',
                    help='root_directory in which outputs are saved')
parser.add_argument("--res_dir", type=str,
                    help='directory in which exp. res are saved')
parser.add_argument("--num_it", type=int, default=300000,
                    help='iteration number')
parser.add_argument("--net_archi", default='conv',
                    help='networks architecture [mlp/conv]')
parser.add_argument("--batch_size", type=int,
                    help='batch size')
parser.add_argument("--lr", type=float,
                    help='learning rate size')
parser.add_argument("--lr_decay", action='store_true', default=False,
                    help='learning rate decay')
parser.add_argument("--beta", type=float, default=0.,
                    help='beta')
parser.add_argument("--gamma", type=float, default=1.,
                    help='weight for mass reg. in ground cost')
parser.add_argument("--id", type=int, default=0,
                    help='exp. config. id')
parser.add_argument("--sigma_pen", action='store_true', default=False,
                    help='penalization of Sigma_q')
parser.add_argument("--sigma_pen_val", type=float, default=0.01,
                    help='value of penalization of Sigma_q')
parser.add_argument("--cost", default='l2sq',
                    help='ground cost')
parser.add_argument('--save_model', action='store_false', default=True,
                    help='save final model weights [True/False]')
parser.add_argument("--save_data", action='store_false', default=True,
                    help='save training data')
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
    if FLAGS.dataset == 'mnist':
        opts = configs.config_mnist
        opts['zdim'] = 16
    elif FLAGS.dataset == 'svhn':
        opts = configs.config_svhn
        opts['zdim'] = 16
    elif FLAGS.dataset == 'cifar10':
        opts = configs.config_cifar10
        opts['zdim'] = 128
    elif FLAGS.dataset == 'celebA':
        opts = configs.config_celeba
        opts['zdim'] = 64
    else:
        assert False, 'Unknown dataset'
    # set data_dir
    if FLAGS.data_dir:
        opts['data_dir'] = FLAGS.data_dir
    else:
        raise Exception('You must provide a data_dir')

    ## exp conf
    betas = [0.1,1,5,10,25,50,75,100]
    coef_id = (FLAGS.id-1) % len(betas)
    # coef_id = (FLAGS.id-1) % len(exp_config)
    # exp_config = list(itertools.product(lr_decay,gammas, orientations))

    ## Set method param
    opts['pen_enc_sigma'] = FLAGS.sigma_pen
    opts['beta_pen_enc_sigma'] = FLAGS.sigma_pen_val

    ## Model set up
    opts['model'] = FLAGS.model
    opts['decoder'] = FLAGS.decoder
    opts['net_archi'] = FLAGS.net_archi
    if opts['model'][-3:]=='VAE':
        opts['input_normalize_sym'] = False
    if FLAGS.batch_size:
        opts['batch_size'] = FLAGS.batch_size
    if FLAGS.lr:
        opts['lr'] = FLAGS.lr
    opts['lr_decay'] = FLAGS.lr_decay
    opts['beta'] = betas[coef_id]
    # opts['beta'] = FLAGS.beta

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
    if not tf.io.gfile.isdir(results_dir):
        utils.create_dir(results_dir)
    opts['out_dir'] = os.path.join(results_dir,FLAGS.out_dir)
    if not tf.io.gfile.isdir(opts['out_dir']):
        utils.create_dir(opts['out_dir'])
    out_subdir = os.path.join(opts['out_dir'], opts['model'])
    if not tf.io.gfile.isdir(out_subdir):
        utils.create_dir(out_subdir)
    out_subsubdir = os.path.join(out_subdir, opts['cost']) # + '_' + str(int((FLAGS.id-1) / len(betas))))
    if not tf.io.gfile.isdir(out_subsubdir):
        utils.create_dir(out_subsubdir)
    exp_name = 'beta_' + str(opts['beta'])
    if opts['cost']=='sw':
        exp_name += '_' + opts['sw_proj_type'] + '_L' + str(opts['sw_proj_num'])
        if opts['sw_proj_type']=='max-sw' or opts['sw_proj_type']=='max-gsw':
            exp_name += '_dfreq' + str(opts['d_updt_freq']) + '_dit' + str(opts['d_updt_it'])
    elif opts['cost'][:4]=='wgan':
        exp_name += '_gamma_' + str(opts['gamma'])
        # critic archi
        exp_name += '_' + opts['wgan_critic_archi']
        # critic training setup
        exp_name += '_dfreq_' + str(opts['d_updt_freq'])
        # critic reg
        exp_name += '_dit_' + str(opts['d_updt_it'])
        # critic reg
        exp_name += '_l_' + str(opts['lambda'])
    elif opts['cost']=='wemd':
        exp_name += '_gamma_' + str(opts['gamma'])
        exp_name += '_L_' + str(opts['orientation_num'])
    if FLAGS.res_dir:
        exp_name += '_' + FLAGS.res_dir
    opts['exp_dir'] = os.path.join(out_subsubdir, exp_name)
    if not tf.io.gfile.isdir(opts['exp_dir']):
        utils.create_dir(opts['exp_dir'])
        utils.create_dir(os.path.join(opts['exp_dir'], 'checkpoints'))

    # Verbose
    logging.basicConfig(filename=os.path.join(opts['exp_dir'],'outputs.log'),
        level=logging.INFO, format='%(asctime)s - %(message)s')

    # Loading the dataset
    data = DataHandler(opts)
    assert data.train_size >= opts['batch_size'], 'Training set too small'

    opts['it_num'] = FLAGS.num_it
    opts['print_every'] = int(opts['it_num'] / 10.)
    opts['evaluate_every'] = int(opts['it_num'] / 20.)
    opts['save_every'] = 10000000000
    opts['save_final'] = FLAGS.save_model
    opts['save_train_data'] = FLAGS.save_data
    opts['vizu_encSigma'] = False
    opts['fid'] = True

    #Reset tf graph
    tf.compat.v1.reset_default_graph()

    # Loading the dataset
    data = DataHandler(opts)
    assert data.train_size >= opts['batch_size'], 'Training set too small'

    # inti method
    run = Run(opts, data)

    # Training/testing/vizu
    if FLAGS.mode=="train":
        # Dumping all the configs to the text file
        with utils.o_gfile((opts['exp_dir'], 'params.txt'), 'w') as text:
            text.write('Parameters:\n')
            for key in opts:
                text.write('%s : %s\n' % (key, opts[key]))
        run.train()
    else:
        assert False, 'Unknown mode %s' % FLAGS.mode

main()
