# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class helps to handle the data.

"""

import os
import shutil
import random
import logging
import gzip
import zipfile
import tensorflow as tf
import numpy as np
from six.moves import cPickle
import urllib.request
import requests
from scipy.io import loadmat
from sklearn.feature_extraction import image
import struct
from tqdm import tqdm
from PIL import Image
import sys
import tarfile
import h5py
from math import ceil

import utils

import pdb

# Path to data
data_dir = '../../data'

datashapes = {}
datashapes['mnist'] = [32, 32, 1]
datashapes['svhn'] = [32, 32, 3]
datashapes['cifar10'] = [32, 32, 3]
datashapes['celebA'] = [64, 64, 3]


def _data_dir(opts):
    _data_dir = os.path.join(data_dir, opts['dataset'])
    if opts['dataset']=='mnist':
        data_path = _data_dir
    elif opts['dataset']=='svhn':
        data_path = _data_dir
    elif opts['dataset']=='cifar10':
        data_path = os.path.join(_data_dir, 'cifar-10-batches-py')
    elif opts['dataset']=='celebA':
        data_path = os.path.join(_data_dir,'img_align_celeba')
    else:
        assert False, 'Unknow {} dataset'.format(opts['dataset'])
    assert os.path.isdir(data_path), '{} dir. doesnt exist'. format(opts['dataset'])
    return data_path

def _load_cifar_batch(fpath):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.

    # Returns
        A ndarray data.
    """
    f = utils.o_gfile(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']

    data = data.reshape(data.shape[0], 3, 32, 32).astype(dtype=np.float32)
    return data


class DataHandler(object):
    """A class storing and manipulating the dataset.

    In this code we asume a data point is a 3-dimensional array, for
    instance a 28*28 grayscale picture would correspond to (28,28,1),
    a 16*16 picture of 3 channels corresponds to (16,16,3) and a 2d point
    corresponds to (2,1,1). The shape is contained in self.data_shape
    """


    def __init__(self, opts):
        self.dataset = opts['dataset']
        self.crop_style = opts['celebA_crop']
        # load data
        logging.error('\n Loading {}.'.format(self.dataset))
        self._init_dataset(opts)
        logging.error('Loading Done.')

    def _init_dataset(self, opts):
        """Load a dataset and fill all the necessary variables.

        """
        if self.dataset == 'mnist':
            self._load_mnist(opts)
        elif self.dataset == 'svhn':
            self._load_svhn(opts)
        elif self.dataset == 'cifar10':
            self._load_cifar10(opts)
        elif self.dataset == 'celebA':
            self._load_celebA(opts)
        else:
            raise ValueError('Unknown {} dataset' % self.dataset)

    def _load_mnist(self, opts):
        """Load data from MNIST or ZALANDO files.

        """
        self.data_dir = _data_dir(opts)
        # loading images
        tr_X, te_X = None, None
        with gzip.open(os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(60000*28*28*1), dtype=np.uint8)
            tr_X = loaded.reshape((60000, 28, 28, 1)).astype(np.float32)
        with gzip.open(os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(10000*28*28*1), dtype=np.uint8)
            te_X = loaded.reshape((10000, 28, 28, 1)).astype(np.float32)
        X = np.concatenate((tr_X, te_X), axis=0)
        self.all_data = X / 255.
        num_data = X.shape[0]
        # plot set
        idx = np.random.randint(self.all_data.shape[0],size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-10000:
            tr_stop = num_data - 10000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-10000:]]
        # dataset size
        self.train_size = data_train.shape[0]
        self.test_size = data_test.shape[0]
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # pad data to 32x32
        def pad_mnist(x):
            paddings = [[2,2], [2,2], [0,0]]
            return tf.pad(x, paddings, mode='CONSTANT', constant_values=0.)
        dataset_train = dataset_train.map(pad_mnist,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_test = dataset_test.map(pad_mnist,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # normalize data if needed
        if opts['input_normalize_sym']:
            dataset_train = dataset_train.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.data_vizu = (self.data_vizu - 0.5) * 2.
            self.data_plot = (self.data_plot - 0.5) * 2.
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = dataset_train.make_initializable_iterator()
        self.iterator_test = dataset_test.make_initializable_iterator()

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
            self.handle, dataset_train.output_types, dataset_train.output_shapes).get_next()

    def _load_svhn(self, opts):
        """Load data from SVHN files.

        """
        # Helpers to process raw data
        def convert_imgs_to_array(img_array):
            rows = datashapes['svhn'][0]
            cols = datashapes['svhn'][1]
            chans = datashapes['svhn'][2]
            num_imgs = img_array.shape[3]
            # Note: not the most efficent way but can monitor what is happening
            new_array = np.empty(shape=(num_imgs, rows, cols, chans), dtype=np.float32)
            for x in range(0, num_imgs):
                # TODO reuse normalize_img here
                chans = img_array[:, :, :, x]
                # # normalize pixels to 0 and 1. 0 is pure white, 1 is pure channel color
                # norm_vec = (255-chans)*1.0/255.0
                new_array[x] = chans
            return new_array

        self.data_dir = _data_dir(opts)
        # loading images
        # Training data
        file_path = os.path.join(self.data_dir,'train_32x32.mat')
        file = open(file_path, 'rb')
        data = loadmat(file)
        imgs = data['X']
        tr_X = convert_imgs_to_array(imgs)
        file.close()
        if opts['use_extra']:
            file_path = os.path.join(self.data_dir,'extra_32x32.mat')
            file = open(file_path, 'rb')
            data = loadmat(file)
            imgs = data['X']
            extra_X = convert_imgs_to_array(imgs)
            extra_X = extra_X
            file.close()
            # concatenate training and extra
            tr_X = np.concatenate((tr_X,extra_X), axis=0)
        # Testing data
        file_path = os.path.join(self.data_dir,'test_32x32.mat')
        file = open(file_path, 'rb')
        data = loadmat(file)
        imgs = data['X']
        te_X = convert_imgs_to_array(imgs)
        file.close()
        X = np.concatenate((tr_X, te_X), axis=0)
        self.all_data = X / 255.
        num_data = X.shape[0]
        # plot set
        idx = np.random.randint(self.all_data.shape[0],size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-10000:
            tr_stop = num_data - 10000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-10000:]]
        # dataset size
        self.train_size = data_train.shape[0]
        self.test_size = data_test.shape[0]
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # normalize data if needed
        if opts['input_normalize_sym']:
            dataset_train = dataset_train.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.data_vizu = (self.data_vizu - 0.5) * 2.
            self.data_plot = (self.data_plot - 0.5) * 2.
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = dataset_train.make_initializable_iterator()
        self.iterator_test = dataset_test.make_initializable_iterator()

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
            self.handle, dataset_train.output_types, dataset_train.output_shapes).get_next()

    def _load_cifar10(self, opts, ):
        """Load data from MNIST or ZALANDO files.

        """

        self.data_dir = _data_dir(opts)
        # loading data
        num_train_samples = 50000
        data_dir = _data_dir(opts)
        x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
        for i in range(1, 6):
            fpath = os.path.join(self.data_dir, 'data_batch_' + str(i))
            data = _load_cifar_batch(fpath)
            x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        x_train = x_train.transpose(0, 2, 3, 1)
        fpath = os.path.join(self.data_dir, 'test_batch')
        x_test = _load_cifar_batch(fpath)
        x_test = x_test.transpose(0, 2, 3, 1)
        X = np.vstack([x_train, x_test])
        self.all_data = X / 255.
        num_data = X.shape[0]
        # plot set
        idx = np.random.randint(self.all_data.shape[0],size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-10000:
            tr_stop = num_data - 10000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-10000:]]
        # dataset size
        self.train_size = data_train.shape[0]
        self.test_size = data_test.shape[0]
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # normalize data if needed
        if opts['input_normalize_sym']:
            dataset_train = dataset_train.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.data_vizu = (self.data_vizu - 0.5) * 2.
            self.data_plot = (self.data_plot - 0.5) * 2.
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = dataset_train.make_initializable_iterator()
        self.iterator_test = dataset_test.make_initializable_iterator()

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
            self.handle, dataset_train.output_types, dataset_train.output_shapes).get_next()

    def _load_celebA(self, opts):
        """Load CelebA
        """
        num_data = 202599
        self.data_dir = _data_dir(opts)
        self.all_data = np.array([os.path.join(self.data_dir,'%.6d.jpg') % i for i in range(1, num_data + 1)])
        # plot set
        idx = np.random.randint(self.all_data.shape[0],size=opts['evaluate_num_pics'])
        self.data_plot = self._sample_observations(idx)
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(num_data)
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>num_data-10000:
            tr_stop = num_data - 10000
        else:
            tr_stop = opts['train_dataset_size']
        data_train = self.all_data[idx_random[:tr_stop]]
        data_test = self.all_data[idx_random[-10000:]]
        # dataset size
        self.train_size = data_train.shape[0]
        self.test_size = data_test.shape[0]
        # data for vizualisation
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(self.test_size, size=opts['plot_num_pics'])
        self.data_vizu = self._sample_observations(idx)
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(data_test)
        # map files paths to image with tf.io.decode_jpeg
        def process_path(file_path):
            # reading .jpg file
            image_file = tf.read_file(file_path)
            im_decoded = tf.cast(tf.image.decode_jpeg(image_file, channels=3), dtype=tf.dtypes.float32)
            # crop and resize
            width = 178
            height = 218
            new_width = 140
            new_height = 140
            if self.crop_style == 'closecrop':
                # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
                left = (width - new_width) / 2
                top = (height - new_height) / 2
                right = (width + new_width) / 2
                bottom = (height + new_height) / 2
                im = tf.image.crop_and_resize(tf.expand_dims(im_decoded,axis=0),
                                        np.array([[top / (height-1), right / (width-1), bottom / (height-1), left / (width-1)]]),
                                        [0,],
                                        (64,64),
                                        method='bilinear', extrapolation_value=0)
            # elif self.crop_style == 'resizecrop':
            #     # This method was used in ALI, AGE, ...
            #     im = im.resize((64, 78), Image.ANTIALIAS)
            #     im = im.crop((0, 7, 64, 64 + 7))
            else:
                assert False, '{} not implemented.'.format(self.crop_style)
            return tf.reshape(im, datashapes['celebA']) / 255.
        dataset_train = dataset_train.map(process_path,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_test = dataset_test.map(process_path,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # normalize data if needed
        if opts['input_normalize_sym']:
            dataset_train = dataset_train.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.map(lambda x: (x - 0.5) * 2.,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.data_vizu = (self.data_vizu - 0.5) * 2.
            self.data_plot = (self.data_plot - 0.5) * 2.
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = dataset_train.make_initializable_iterator()
        self.iterator_test = dataset_test.make_initializable_iterator()

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
            self.handle, dataset_train.output_types, dataset_train.output_shapes).get_next()

    def init_iterator(self, sess):
        sess.run([self.iterator_train.initializer,self.iterator_test.initializer])
        # handle = sess.run(iterator.string_handle())
        train_handle, test_handle = sess.run([self.iterator_train.string_handle(),self.iterator_test.string_handle()])

        return train_handle, test_handle

    def _sample_observations(self, keys):
        if len(self.all_data.shape)>1:
            # all_data is an np.ndarray already loaded into the memory
            if self.dataset=='mnist':
                obs = self.all_data[keys]
                paddings = ((0,0),(2,2), (2,2), (0,0))
                obs = np.pad(obs, paddings, mode='constant', constant_values=0.)
            else:
                obs = self.all_data[keys]
            return obs
        else:
            # all_data is a 1d array of paths
            obs = []
            keys = list(keys)
            for key in keys:
                img = self._read_image(key)
                obs.append(img)
            return np.stack(obs)

    def _read_image(self, key):
        seed = 123
        assert key==int(os.path.split(self.all_data[key])[1][:-4])-1, 'Mismatch between key and img_file_name'
        if self.dataset == 'celebA':
            point = self._read_celeba_image(self.all_data[key])
        else:
            raise Exception('Disc read for {} not implemented yet...'.format(self.dataset))

        return point

    def _read_celeba_image(self, file_path):
        width = 178
        height = 218
        new_width = 140
        new_height = 140
        im = Image.open(file_path)
        if self.crop_style == 'closecrop':
            # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height)/2
            im = im.crop((left, top, right, bottom))
            im = im.resize((64, 64), Image.ANTIALIAS)
        elif self.crop_style == 'resizecrop':
            # This method was used in ALI, AGE, ...
            im = im.resize((64, 78), Image.ANTIALIAS)
            im = im.crop((0, 7, 64, 64 + 7))
        else:
            raise Exception('Unknown crop style specified')
        im_array = np.array(im).reshape(datashapes['celebA']) / 255.
        im.close()
        return im_array
