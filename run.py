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

import pdb

parser = argparse.ArgumentParser()
# Args for experiment
parser.add_argument("--model", default='TCWAE_MWS',
                    help='model to train [WAE/BetaVAE/...]')
parser.add_argument("--mode", default='train',
                    help='mode to run [train/vizu/fid/test]')
parser.add_argument("--exp", default='dsprites',
                    help='dataset [mnist/cifar10/].'\
                    ' celebA/dsprites Not implemented yet')
parser.add_argument("--data_dir", type=str, default='../data',
                    help='directory in which data is stored')
parser.add_argument("--out_dir", type=str, default='code_outputs',
                    help='root_directory in which outputs are saved')
parser.add_argument("--exp_dir", type=str, default='res',
                    help='directory in which exp. outputs are saved')
parser.add_argument("--num_it", type=int, default=300000,
                    help='iteration number')
parser.add_argument("--net_archi",
                    help='networks architecture [mlp/dcgan/resnet]')
parser.add_argument("--idx", type=int, default=0,
                    help='idx latent reg weight setup')
parser.add_argument("--sigma_pen", default='False',
                    help='penalization of Sigma_q')
parser.add_argument("--sp", type=float, default=0.01,
                    help='value of penalization of Sigma_q')
parser.add_argument("--cost", default='l2sq',
                    help='ground cost [l2, l2sq, l2sq_norm, l1, xentropy, emd]')
parser.add_argument("--emd_cost", default='l2sq',
                    help='ground cost of emd cost [l2, l2sq, l2sq_norm, l1, xentropy]')
parser.add_argument("--sink_reg", type=float, default=0.01,
                    help='sinkhorn regularization param.')
parser.add_argument("--L", type=int, default=100,
                    help='sinkhorn iterations num.')
parser.add_argument("--vizu_sinkit", default='True',
                    help='plot and vizu sinkhorn iterations.')
parser.add_argument("--save_model",
                    help='save final model weights [True/False]')
parser.add_argument("--save_data", default='True',
                    help='save training data [True/False]')
parser.add_argument("--weights_file")


FLAGS = parser.parse_args()


# --- Network architectures
mlp_config = { 'e_arch': 'mlp' , 'e_nlayers': 2, 'e_nfilters': [2048, 2048], 'e_nonlinearity': 'relu',
        'd_arch': 'mlp' , 'd_nlayers': 2, 'd_nfilters': [2048, 2048], 'd_nonlinearity': 'relu'}
dcgan_config = { 'e_arch': 'dcgan' , 'e_nlayers': 4, 'e_nfilters': [32,64,128,256], 'e_nonlinearity': 'relu',
        'd_arch': 'dcgan' , 'd_nlayers': 4, 'd_nfilters': [32,64,128,256], 'd_nonlinearity': 'relu',
        'filter_size': [4,4,4,4]}

net_configs = {'mlp': mlp_config, 'dcgan': dcgan_config}


def main():

    # Select dataset to use
    if FLAGS.exp == 'dsprites':
        opts = configs.config_dsprites
    elif FLAGS.exp == 'smallNORB':
        opts = configs.config_smallNORB
    elif FLAGS.exp == '3dshapes':
        opts = configs.config_3dshapes
    elif FLAGS.exp == '3Dchairs':
        opts = configs.config_3Dchairs
    elif FLAGS.exp == 'celebA':
        opts = configs.config_celebA
    elif FLAGS.exp == 'mnist':
        opts = configs.config_mnist
    else:
        assert False, 'Unknown experiment dataset'

    # Select training method
    if FLAGS.model:
        opts['model'] = FLAGS.model

    # Data directory
    opts['data_dir'] = FLAGS.data_dir

    # Mode
    if FLAGS.mode=='fid':
        opts['fid'] = True
    else:
        opts['fid'] = False

    # Obj set up
    opts['cost'] = FLAGS.cost #l2, l2sq, l2sq_norm, l1, xentropy
    if FLAGS.sink_reg:
        opts['sinkhorn_reg'] = FLAGS.sink_reg
    if FLAGS.L:
        opts['sinkhorn_iterations'] = FLAGS.L
    if FLAGS.exp == 'mnist':
        beta = [1, 2, 5, 10, 20, 50, 100]
    elif FLAGS.exp == 'celebA':
        beta = [1, 10, 20, 50, 100, 200, 500]
    else:
        assert False, 'Unknow {} dataset' % FLAGS.exp
    opts['beta'] = beta[FLAGS.idx-1]
    # Penalty Sigma_q
    opts['pen_enc_sigma'] = FLAGS.sigma_pen=='True'
    opts['lambda_pen_enc_sigma'] = FLAGS.sp

    # NN set up
    if FLAGS.net_archi:
        opts['net_archi'] = FLAGS.net_archi
    opts['network'] = net_configs[opts['net_archi']]

    # Create directories
    if FLAGS.out_dir:
        opts['out_dir'] = FLAGS.out_dir
    if FLAGS.exp_dir:
        opts['exp_dir'] = FLAGS.exp_dir
    exp_dir = os.path.join(opts['out_dir'],
                           opts['model'],
                           '{}_{}_{:%Y_%m_%d_%H_%M}'.format(
                                opts['exp_dir'],
                                opts['beta'],
                                datetime.now()), )
    opts['exp_dir'] = exp_dir
    if not tf.gfile.IsDirectory(exp_dir):
        utils.create_dir(exp_dir)
        utils.create_dir(os.path.join(exp_dir, 'checkpoints'))

    # Verbose
    logging.basicConfig(filename=os.path.join(exp_dir,'outputs.log'),
        level=logging.INFO, format='%(asctime)s - %(message)s')

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    # Experiemnts set up
    opts['epoch_num'] = int(FLAGS.num_it / int(data.num_points/opts['batch_size']))
    opts['print_every'] = int(opts['epoch_num'] / 10.) * int(data.num_points/opts['batch_size'])-1
    opts['evaluate_every'] = int(opts['print_every'] / 2.) + 1
    opts['save_every'] = 1000000000
    if FLAGS.vizu_sinkit:
        opts['vizu_sinkhorn'] = FLAGS.vizu_sinkit
    if FLAGS.save_model=='True':
        opts['save_final'] = True
    else:
        opts['save_final'] = False
    if FLAGS.save_data=='True':
        opts['save_train_data'] = True
    else:
        opts['save_train_data'] = False
    opts['vizu_encSigma'] = False


    #Reset tf graph
    tf.reset_default_graph()

    run = Run(opts)

    # Training/testing/vizu
    if FLAGS.mode=="train":
        # Dumping all the configs to the text file
        with utils.o_gfile((exp_dir, 'params.txt'), 'w') as text:
            text.write('Parameters:\n')
            for key in opts:
                text.write('%s : %s\n' % (key, opts[key]))
        run.train(data, FLAGS.weights_file)
    else:
        assert False, 'To implement.'

main()
