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
import struct
from tqdm import tqdm
from PIL import Image
import sys
import tarfile
import h5py

import utils

import pdb

datashapes = {}
datashapes['celebA'] = [64, 64, 3]
datashapes['mnist'] = [28, 28, 1]
datashapes['svhn'] = [32, 32, 3]
datashapes['cifar'] = [32, 32, 3]


def _data_dir(opts):
    data_path = maybe_download(opts)
    return data_path

def maybe_download(opts):
    """Download the data from url, unless it's already here."""
    if not tf.gfile.Exists(opts['data_dir']):
        tf.gfile.MakeDirs(opts['data_dir'])
    data_path = os.path.join(opts['data_dir'], opts['dataset'])
    if not tf.gfile.Exists(data_path):
        tf.gfile.MakeDirs(data_path)
    if opts['dataset']=='mnist':
        maybe_download_file(data_path,'train-images-idx3-ubyte.gz',opts['MNIST_data_source_url'])
        maybe_download_file(data_path,'train-labels-idx1-ubyte.gz',opts['MNIST_data_source_url'])
        maybe_download_file(data_path,'t10k-images-idx3-ubyte.gz',opts['MNIST_data_source_url'])
        maybe_download_file(data_path,'t10k-labels-idx1-ubyte.gz',opts['MNIST_data_source_url'])
    elif opts['dataset']=='zalando':
        maybe_download_file(data_path,'train-images-idx3-ubyte.gz',opts['Zalando_data_source_url'])
        maybe_download_file(data_path,'train-labels-idx1-ubyte.gz',opts['Zalando_data_source_url'])
        maybe_download_file(data_path,'t10k-images-idx3-ubyte.gz',opts['Zalando_data_source_url'])
        maybe_download_file(data_path,'t10k-labels-idx1-ubyte.gz',opts['Zalando_data_source_url'])
    elif opts['dataset']=='svhn':
        maybe_download_file(data_path,'train_32x32.mat',opts['SVHN_data_source_url'])
        maybe_download_file(data_path,'test_32x32.mat',opts['SVHN_data_source_url'])
        if opts['use_extra']:
            maybe_download_file(data_path,'extra_32x32.mat',opts['SVHN_data_source_url'])
    elif opts['dataset']=='cifar10':
        maybe_download_file(data_path,'cifar-10-python.tar.gz',opts['cifar10_data_source_url'])
        tar = tarfile.open(os.path.join(data_path,'cifar-10-python.tar.gz'))
        tar.extractall(path=data_path)
        tar.close()
        data_path = os.path.join(data_path,'cifar-10-batches-py')
    elif opts['dataset']=='celebA':
        filename = 'img_align_celeba'
        file_path = os.path.join(data_path, filename)
        if not tf.gfile.Exists(file_path):
            filename = 'img_align_celeba.zip'
            file_path = os.path.join(data_path, filename)
            if not tf.gfile.Exists(file_path):
                assert False, '{} dataset does not exist'.format(opts['dataset'])
                download_file_from_google_drive(file_path,filename,opts['celebA_data_source_url'])
            # Unzipping
            print('Unzipping celebA...')
            with zipfile.ZipFile(file_path) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(data_path)
            print('Unzipping done.')
            os.remove(file_path)
            # os.rename(os.path.join(data_path, zip_dir), os.path.join(data_path, 'img_align_celeba'))
        data_path = os.path.join(data_path,'img_align_celeba')
    else:
        assert False, 'Unknow dataset'

    return data_path

def maybe_download_file(name,filename,url):
    filepath = os.path.join(name, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(url + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')

def load_cifar_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
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
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

def transform_mnist(pic, mode='n'):
    """Take an MNIST picture normalized into [0, 1] and transform
        it according to the mode:
        n   -   noise
        i   -   colour invert
        s*  -   shift
    """
    pic = np.copy(pic)
    if mode == 'n':
        noise = np.random.randn(28, 28, 1)
        return np.clip(pic + 0.25 * noise, 0, 1)
    elif mode == 'i':
        return 1. - pic
    pixels = 3 + np.random.randint(5)
    if mode == 'sl':
        pic[:, :-pixels] = pic[:, pixels:] + 0.0
        pic[:, -pixels:] = 0.
    elif mode == 'sr':
        pic[:, pixels:] = pic[:, :-pixels] + 0.0
        pic[:, :pixels] = 0.
    elif mode == 'sd':
        pic[pixels:, :] = pic[:-pixels, :] + 0.0
        pic[:pixels, :] = 0.
    elif mode == 'su':
        pic[:-pixels, :] = pic[pixels:, :] + 0.0
        pic[-pixels:, :] = 0.
    return pic


class Data(object):
    """
    If the dataset can be quickly loaded to memory self.X will contain np.ndarray
    Otherwise we will be reading files as we train. In this case self.X is a structure:
        self.X.paths        list of paths to the files containing pictures
        self.X.dict_loaded  dictionary of (key, val), where key is the index of the
                            already loaded datapoint and val is the corresponding index
                            in self.X.loaded
        self.X.loaded       list containing already loaded pictures
    """
    def __init__(self, opts, X, paths=None, dict_loaded=None, loaded=None):
        """
        X is either np.ndarray or paths
        """
        data_dir = _data_dir(opts)
        self.X = None
        self.normalize = opts['input_normalize_sym']
        self.paths = None
        self.dict_loaded = None
        self.loaded = None
        if isinstance(X, np.ndarray):
            self.X = X
            self.shape = X.shape
        else:
            assert isinstance(data_dir, str), 'Data directory not provided'
            assert paths is not None and len(paths) > 0, 'No paths provided for the data'
            self.data_dir = data_dir
            self.paths = paths[:]
            self.dict_loaded = {} if dict_loaded is None else dict_loaded
            self.loaded = [] if loaded is None else loaded
            self.crop_style = opts['celebA_crop']
            self.dataset_name = opts['dataset']
            self.shape = (len(self.paths), None, None, None)

    def __len__(self):
        if isinstance(self.X, np.ndarray):
            return len(self.X)
        else:
            # Our dataset was too large to fit in the memory
            return len(self.paths)

    def drop_loaded(self):
        if not isinstance(self.X, np.ndarray):
            self.dict_loaded = {}
            self.loaded = []

    def __getitem__(self, key):
        if isinstance(self.X, np.ndarray):
            return self.X[key]
        else:
            # Our dataset was too large to fit in the memory
            if isinstance(key, int):
                keys = [key]
            elif isinstance(key, list):
                keys = key
            elif isinstance(key, np.ndarray):
                keys = list(key)
            elif isinstance(key, slice):
                start = key.start
                stop = key.stop
                step = key.step
                start = start if start is not None else 0
                if start < 0:
                    start += len(self.paths)
                stop = stop if stop is not None else len(self.paths) - 1
                if stop < 0:
                    stop += len(self.paths)
                step = step if step is not None else 1
                keys = range(start, stop, step)
            else:
                print(type(key))
                raise Exception('This type of indexing yet not supported for the dataset')
            res = []
            new_keys = []
            new_points = []
            for key in keys:
                if key in self.dict_loaded:
                    idx = self.dict_loaded[key]
                    res.append(self.loaded[idx])
                else:
                    if self.dataset_name == 'celebA':
                        point = self._read_celeba_image(self.data_dir, self.paths[key])
                    else:
                        raise Exception('Disc read for this dataset not implemented yet...')
                    if self.normalize:
                        point = (point - 0.5) * 2.
                    res.append(point)
                    new_points.append(point)
                    new_keys.append(key)
            n = len(self.loaded)
            cnt = 0
            for key in new_keys:
                self.dict_loaded[key] = n + cnt
                cnt += 1
            self.loaded.extend(new_points)
            return np.array(res)

    def _read_celeba_image(self, data_dir, filename):
        width = 178
        height = 218
        new_width = 140
        new_height = 140
        im = Image.open(utils.o_gfile((data_dir, filename), 'rb'))
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
        return np.array(im).reshape(64, 64, 3) / 255.

class DataHandler(object):
    """A class storing and manipulating the dataset.

    In this code we asume a data point is a 3-dimensional array, for
    instance a 28*28 grayscale picture would correspond to (28,28,1),
    a 16*16 picture of 3 channels corresponds to (16,16,3) and a 2d point
    corresponds to (2,1,1). The shape is contained in self.data_shape
    """


    def __init__(self, opts):
        self.data_shape = None
        self.num_points = None
        self.data = None
        self.test_data = None
        self.labels = None
        self.test_labels = None
        self._load_data(opts)

    def _load_data(self, opts):
        """Load a dataset and fill all the necessary variables.

        """
        if opts['dataset'] == 'celebA':
            self._load_celebA(opts)
        elif opts['dataset'] == 'mnist':
            self._load_mnist(opts)
        elif opts['dataset'] == 'svhn':
            self._load_svhn(opts)
        elif opts['dataset'] == 'cifar':
            self._load_cifar(opts)
        else:
            raise ValueError('Unknown %s' % opts['dataset'])

        if opts['input_normalize_sym']:
            # Normalize data to [-1, 1]
            if isinstance(self.data.X, np.ndarray):
                self.data.X = (self.data.X - 0.5) * 2.
                self.test_data.X = (self.test_data.X - 0.5) * 2.
            # Else we will normalyze while reading from disk

    def _load_celebA(self, opts):
        """Load CelebA
        """
        logging.error('Loading CelebA dataset')

        num_samples = 202599

        paths = np.array(['%.6d.jpg' % i for i in range(1, num_samples + 1)])
        # Creating shuffling mask
        seed = 123
        shuffling_mask = np.arange(num_samples)
        np.random.seed(seed)
        np.random.shuffle(shuffling_mask)
        np.random.seed()
        np.random.shuffle(shuffling_mask[opts['plot_num_pics']:])
        # training set
        self.data = Data(opts, None, paths[shuffling_mask[:-10000]])
        # testing set
        self.test_data = Data(opts, None, paths[shuffling_mask[-10000:-opts['plot_num_pics']]])
        # vizu set
        self.vizu_data = Data(opts, None, paths[shuffling_mask[-opts['plot_num_pics']:]])
        # plot set
        # idx = [5,6,14,17,39,50,60,70,80,90]
        idx = np.arange(5,5+50)
        plot_data = Data(opts, None, paths[idx])
        self.plot_data = plot_data[np.arange(50)]
        # data informations
        self.data_shape = datashapes[opts['dataset']]
        self.num_points = len(self.data)

        logging.error('Loading Done.')

    def _load_mnist(self, opts, zalando=False, modified=False):
        """Load data from MNIST or ZALANDO files.

        """
        if zalando:
            logging.error('Loading Fashion MNIST')
        elif modified:
            logging.error('Loading modified MNIST')
        else:
            logging.error('Loading MNIST')
        data_dir = _data_dir(opts)
        with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(60000*28*28*1), dtype=np.uint8)
            tr_X = loaded.reshape((60000, 28, 28, 1)).astype(np.float32)

        with gzip.open(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(60000), dtype=np.uint8)
            tr_Y = loaded.reshape((60000)).astype(np.int)

        with gzip.open(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(10000*28*28*1), dtype=np.uint8)
            te_X = loaded.reshape((10000, 28, 28, 1)).astype(np.float32)

        with gzip.open(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(10000), dtype=np.uint8)
            te_Y = loaded.reshape((10000)).astype(np.int)

        tr_Y = np.asarray(tr_Y)
        te_Y = np.asarray(te_Y)

        X = np.concatenate((tr_X, te_X), axis=0)
        Y = np.concatenate((tr_Y, te_Y), axis=0)
        X = X / 255.

        # Creating shuffling mask
        seed = 123
        shuffling_mask = np.arange(len(Y))
        np.random.seed(seed)
        np.random.shuffle(shuffling_mask)
        np.random.seed()
        np.random.shuffle(shuffling_mask[opts['plot_num_pics']:])
        self.data_order_idx = np.argsort(shuffling_mask)
        # training set
        self.data = Data(opts, X[shuffling_mask[:-10000]])
        self.labels = Data(opts, Y[shuffling_mask[:-10000]])
        # testing set
        # test_size = 10000 - opts['plot_num_pics']
        self.test_data = Data(opts, X[shuffling_mask[-10000:-opts['plot_num_pics']]])
        self.test_labels = Data(opts, Y[shuffling_mask[-10000:-opts['plot_num_pics']]])
        # vizu set
        self.vizu_data = Data(opts, X[shuffling_mask[-opts['plot_num_pics']:]])
        self.vizu_labels = Data(opts, Y[shuffling_mask[-opts['plot_num_pics']:]])
        # plot set
        idx = np.arange(20)
        self.plot_data = Data(opts, X[idx])
        # data informations
        self.data_shape = datashapes[opts['dataset']]
        self.num_points = len(self.data)

        logging.error('Loading Done.')

    def _load_svhn(self, opts):
        """Load data from SVHN files.

        """
        NUM_LABELS = 10

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

        # Extracting data
        data_dir = _data_dir(opts)

        # Training data
        file_path = os.path.join(data_dir,'train_32x32.mat')
        file = open(file_path, 'rb')
        data = loadmat(file)
        imgs = data['X']
        labels = data['y'].flatten()
        labels[labels == 10] = 0  # Fix for weird labeling in dataset
        tr_Y = labels
        tr_X = convert_imgs_to_array(imgs)
        tr_X = tr_X / 255.
        file.close()
        if opts['use_extra']:
            file_path = os.path.join(data_dir,'extra_32x32.mat')
            file = open(file_path, 'rb')
            data = loadmat(file)
            imgs = data['X']
            labels = data['y'].flatten()
            labels[labels == 10] = 0  # Fix for weird labeling in dataset
            extra_Y = labels
            extra_X = convert_imgs_to_array(imgs)
            extra_X = extra_X / 255.
            file.close()
            # concatenate training and extra
            tr_X = np.concatenate((tr_X,extra_X), axis=0)
            tr_Y = np.concatenate((tr_Y,extra_Y), axis=0)
        seed = 123
        np.random.seed(seed)
        np.random.shuffle(tr_X)
        np.random.seed(seed)
        np.random.shuffle(tr_Y)
        np.random.seed()

        # Testing data
        file_path = os.path.join(data_dir,'test_32x32.mat')
        file = open(file_path, 'rb')
        data = loadmat(file)
        imgs = data['X']
        labels = data['y'].flatten()
        labels[labels == 10] = 0  # Fix for weird labeling in dataset
        te_Y = labels
        te_X = convert_imgs_to_array(imgs)
        te_X = te_X / 255.
        file.close()

        self.data_shape = (32,32,3)

        self.data = Data(opts, tr_X)
        self.labels = tr_Y
        self.test_data = Data(opts, te_X)
        self.test_labels = te_Y
        self.num_points = len(self.data)

        logging.error('Loading Done: Train size: %d, Test size: %d' % (self.num_points,len(self.test_data)))

    def sample_observations_from_factors(self, dataset, factors):
        if dataset == 'dsprites':
            indices = np.dot(factors, self.factor_bases).astype(dtype=np.int32)
            images, labels = self.sample_from_factor_indices(indices)
        elif dataset == '3dshapes':
            indices = np.dot(factors, self.factor_bases).astype(dtype=np.int32)
            images, labels = self.sample_from_factor_indices(indices)
        elif dataset == 'smallNORB':
            feature_state_space_index = np.array(np.dot(self.Y, self.factor_bases), dtype=np.int32)
            num_total_atoms = np.prod(self.factor_sizes)
            state_space_to_save_space_index = np.zeros(num_total_atoms, dtype=np.int32)
            state_space_to_save_space_index[feature_state_space_index] = np.arange(num_total_atoms)
            state_space_index = np.dot(factors, self.factor_bases).astype(dtype=np.int32)
            indices = state_space_to_save_space_index[state_space_index]
            images, labels = self.sample_from_factor_indices(indices)
        elif dataset == '3Dchairs':
            assert False, 'to do'
        elif dataset == 'celebA':
            assert False, 'to do'
        elif dataset == 'mnist':
            assert False, 'to do'
        elif dataset == 'svhn':
            assert False, 'to do'
        else:
            raise ValueError('Unknown %s' % opts['dataset'])

        return images, labels

    def sample_from_factor_indices(self, indices):
        indices_to_order = self.data_order_idx[indices]
        images = np.zeros([len(indices)]+self.data_shape)
        labels = np.zeros([len(indices),len(self.factor_sizes)])
        for i, idx in enumerate(list(indices_to_order)):
            if idx<self.num_points:
                images[i] = self.data[idx]
                labels[i] = self.labels[idx]
            elif idx>=self.num_points and idx<(self.num_points+len(self.test_data)):
                images[i] = self.test_data[idx-self.num_points]
                labels[i] = self.test_labels[idx-self.num_points]
            else:
                images[i] = self.vizu_data[idx-(self.num_points+len(self.test_data))]
                labels[i] = self.vizu_labels[idx-(self.num_points+len(self.test_data))]

        return images, labels


def matrix_type_from_magic(magic_number):
    """
    Get matrix data type from magic number
    See here: https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/readme for details.
    Parameters
    ----------
    magic_number: tuple
        First 4 bytes read from small NORB files
    Returns
    -------
    element type of the matrix
    """
    convention = {'1E3D4C51': 'single precision matrix',
                  '1E3D4C52': 'packed matrix',
                  '1E3D4C53': 'double precision matrix',
                  '1E3D4C54': 'integer matrix',
                  '1E3D4C55': 'byte matrix',
                  '1E3D4C56': 'short matrix'}
    magic_str = bytearray(reversed(magic_number)).hex().upper()
    return convention[magic_str]

def _parse_smallNORB_header(file_pointer):
    """
    Parse header of small NORB binary file

    Parameters
    ----------
    file_pointer: BufferedReader
        File pointer just opened in a small NORB binary file
    Returns
    -------
    file_header_data: dict
        Dictionary containing header information
    """
    # Read magic number
    magic = struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

    # Read dimensions
    dimensions = []
    num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
    for _ in range(num_dims):
        dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

    file_header_data = {'magic_number': magic,
                        'matrix_type': matrix_type_from_magic(magic),
                        'dimensions': dimensions}
    return file_header_data

def _read_binary_matrix(filename):
    """Reads and returns binary formatted matrix stored in filename."""
    with tf.gfile.GFile(filename, "rb") as f:
        s = f.read()
        magic = int(np.frombuffer(s, "int32", 1))
        ndim = int(np.frombuffer(s, "int32", 1, 4))
        eff_dim = max(3, ndim)
        raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
        dims = []
        for i in range(0, ndim):
            dims.append(raw_dims[i])

        dtype_map = {507333717: "int8",
                    507333716: "int32",
                    507333713: "float",
                    507333715: "double"}
        data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
    data = data.reshape(tuple(dims))
    return data

def _resize_images(integer_images):
    resized_images = np.zeros((integer_images.shape[0], 64, 64))
    for i in range(integer_images.shape[0]):
        image = Image.fromarray(integer_images[i, :, :])
        image = image.resize((64, 64), Image.ANTIALIAS)
        resized_images[i, :, :] = image
    return resized_images.astype(np.float32) / 255.
