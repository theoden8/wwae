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
# Data paths
parser.add_argument("--dataset", default='celebA',
                    help='dataset')
parser.add_argument("--data_dir", type=str, default='../data',
                    help='directory in which data is stored')
parser.add_argument("--out_dir", type=str, default='code_outputs',
                    help='root_directory in which outputs are saved')
parser.add_argument("--res_dir", type=str, default='res',
                    help='directory in which exp. res are saved')
parser.add_argument("--weights_file")
# model set up
parser.add_argument("--model", default='WAE',
                    help='model to train [WAE/BetaVAE/...]')
parser.add_argument("--net_archi", default='conv',
                    help='networks architecture [mlp/conv]')
# fid setup
parser.add_argument("--compute_stats", action='store_true', default=False,
                    help='wether compute the stats of the dataset or not.')
parser.add_argument("--fid_inputs", default='samples',
                    help='inputs to compute FID')

FLAGS = parser.parse_args()

#run_fid.py --dataset mnist --data_dir ../../data --out_dir mnist_test --res_dir res_10.0_2020_09_15_16_06 --weights_file trained-WAE-final-1000 --net_archi mlp --compute_stats --fid_inputs reconstruction 

def main():

    # Select dataset to use
    if FLAGS.dataset == 'mnist':
        opts = configs.config_mnist
        opts['zdim'] = 16
    elif FLAGS.dataset == 'svhn':
        opts = configs.config_svhn
        opts['zdim'] = 16
    elif FLAGS.dataset == 'celebA':
        opts = configs.config_celeba
        opts['zdim'] = 64
    elif FLAGS.dataset == 'cifar10':
        opts = configs.config_cifar10
        opts['zdim'] = 128
    else:
        assert False, 'Unknown dataset'
    opts['data_dir'] = FLAGS.data_dir

    # Set method param
    opts['data_dir'] = FLAGS.data_dir
    opts['fid'] = True
    opts['net_archi'] = FLAGS.net_archi

    # Model set up
    opts['model'] = FLAGS.model

    # Create directories
    results_dir = 'results'
    opts['out_dir'] = os.path.join(results_dir,FLAGS.out_dir)
    out_subdir = os.path.join(opts['out_dir'], opts['model'])
    opts['exp_dir'] = os.path.join(out_subdir, FLAGS.res_dir)
    if not tf.io.gfile.isdir(opts['exp_dir']):
        raise Exception("Experiment doesn't exist!")

    #Reset tf graph
    tf.reset_default_graph()

    # Loading the dataset
    data = DataHandler(opts)
    assert data.train_size >= opts['batch_size'], 'Training set too small'

    # init method
    run = Run(opts, data)

    # get fid
    run.fid_score(opts['exp_dir'], FLAGS.weights_file, FLAGS.compute_stats, FLAGS.fid_inputs)

main()
