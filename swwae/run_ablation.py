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
parser.add_argument("--res_dir", type=str, default='res',
                    help='directory in which exp. res are saved')
parser.add_argument("--num_it", type=int, default=300000,
                    help='iteration number')
parser.add_argument("--net_archi", default='conv',
                    help='networks architecture [mlp/conv]')
parser.add_argument("--beta", default=10.,
                    help='Latent reg weight setup')
parser.add_argument("--id", type=int, default=0,
                    help='exp. config. id')
parser.add_argument("--sigma_pen", action='store_true', default=False,
                    help='penalization of Sigma_q')
parser.add_argument("--sigma_pen_val", type=float, default=0.01,
                    help='value of penalization of Sigma_q')
parser.add_argument("--cost", default='l2sq',
                    help='ground cost [l1, l2, l2sq, l2sq_norm, sw2]')
parser.add_argument('--save_model', action='store_false', default=True,
                    help='save final model weights [True/False]')
parser.add_argument("--save_data", action='store_false', default=True,
                    help='save training data')
parser.add_argument("--weights_file")


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

    # Set method param
    opts['cost'] = FLAGS.cost #l2, l2sq, l2sq_norm, l1, xentropy
    opts['net_archi'] = FLAGS.net_archi
    opts['pen_enc_sigma'] = FLAGS.sigma_pen
    opts['lambda_pen_enc_sigma'] = FLAGS.sigma_pen_val

    # Slicing config
    if FLAGS.dataset=='mnist':
        # slice_dist = ['det', 'gaussian_small_var', 'gaussian_large_var', 'uniform']
        slice_dist = ['det', 'gaussian_small_var', 'gaussian_large_var', 'uniform']
        L_val = [8,16,32,64]
    elif FLAGS.dataset=='celebA':
        slice_dist = ['det', 'gaussian_small_var', 'gaussian_large_var', 'uniform']
        L_val = [8,16,32]
    elif FLAGS.dataset=='cifar10':
        slice_dist = ['det', 'gaussian_small_var', 'gaussian_large_var', 'uniform']
        L_val = [8,16,32,64,]
    else:
        raise Exception('You must provide a data_dir')

    exp_config = list(itertools.product(slice_dist,L_val))
    exp_id = (FLAGS.id-1) % len(exp_config)
    opts['sw_proj_type'] = exp_config[exp_id][0]
    opts['sw_proj_num'] = exp_config[exp_id][1]
    # pdb.set_trace()

    # Model set up
    opts['model'] = FLAGS.model
    opts['decoder'] = FLAGS.decoder
    if opts['model'][-3:]=='VAE':
        opts['input_normalize_sym'] = False
    opts['beta'] = FLAGS.beta

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
    opts['exp_dir'] = opts['sw_proj_type'] + '_' + str(opts['sw_proj_num'])
    exp_dir = os.path.join(out_subdir,
                           '{}_{}_{:%Y_%m_%d_%H_%M}'.format(
                                opts['exp_dir'],
                                opts['beta'],
                                datetime.now()), )
    opts['exp_dir'] = exp_dir
    if not tf.io.gfile.isdir(exp_dir):
        utils.create_dir(exp_dir)
        utils.create_dir(os.path.join(exp_dir, 'checkpoints'))

    # Verbose
    logging.basicConfig(filename=os.path.join(exp_dir,'outputs.log'),
        level=logging.INFO, format='%(asctime)s - %(message)s')

    # Loading the dataset
    data = DataHandler(opts)
    assert data.train_size >= opts['batch_size'], 'Training set too small'

    opts['it_num'] = FLAGS.num_it
    opts['print_every'] = int(opts['it_num'] / 4.)
    opts['evaluate_every'] = int(opts['print_every'] / 2.) + 1
    opts['save_every'] = 10000000000
    opts['save_final'] = FLAGS.save_model
    opts['save_train_data'] = FLAGS.save_data
    opts['vizu_encSigma'] = False

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
        with utils.o_gfile((exp_dir, 'params.txt'), 'w') as text:
            text.write('Parameters:\n')
            for key in opts:
                text.write('%s : %s\n' % (key, opts[key]))
        run.train()
    else:
        assert False, 'Unknown mode %s' % FLAGS.mode

main()
