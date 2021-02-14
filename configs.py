import copy
from math import pow, sqrt

### Default common config
config = {}
# Outputs set up
config['verbose'] = False
config['save_every'] = 1000
config['save_final'] = True
config['save_train_data'] = True
config['print_every'] = 100
config['evaluate_every'] = int(config['print_every'] / 2)
config['vizu_embedded'] = False
config['embedding'] = 'pca' #vizualisation method of the embeddings: pca, umap
config['vizu_encSigma'] = False
config['vizu_interpolation'] = False
config['vizu_rgb_transformed'] = True
config['fid'] = False
config['out_dir'] = 'code_outputs'
config['plot_num_pics'] = 100
config['plot_num_cols'] = 10
config['evaluate_num_pics'] = 5
# Experiment set up
config['train_dataset_size'] = -1
config['epoch_num'] = 101
config['model'] = 'WAE' #WAE, BetaVAE
config['use_trained'] = False #train from pre-trained model
# Data set up
config['celebA_crop'] = 'closecrop' # closecrop, resizecrop
# Opt set up
config['optimizer'] = 'adam' # adam, sgd
config['adam_beta1'] = 0.9
config['adam_beta2'] = 0.999
config['lr'] = 0.001
config['lr_decay'] = False
config['lr_adv'] = 1e-08
config['normalization'] = 'batchnorm' #batchnorm, layernorm, none
config['batch_norm_eps'] = 1e-05
config['batch_norm_momentum'] = 0.99
# Objective set up
config['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1, xentropy
config['mmd_kernel'] = 'IMQ' # RBF, IMQ
config['transform_rgb_img'] = 'none' #[average/wavelength/learned/none]
config['pretrain_critic_nit'] = 10000
config['d_updt_freq'] = 1
config['d_updt_it'] = 5
config['pen_enc_sigma'] = False
config['beta_pen_enc_sigma'] = 0.001
config['sw_proj_num'] = 15
config['sw_proj_type'] = 'det'  # det for deterministic, or 'uniform'
# Model set up
config['pz_scale'] = 1.
config['prior'] = 'gaussian' # dirichlet, gaussian
config['encoder'] = 'gauss' # deterministic, gaussian
config['decoder'] = 'det' # deterministic, gaussian
# beta set up
config['beta'] = 10
config['beta_schedule'] = 'constant' # adaptive, constant
# NN set up
config['init_std'] = 0.099999
config['init_bias'] = 0.0
config['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config['conv_init'] = 'glorot_uniform' #he, glorot, normilized_glorot, truncated_norm

### GMM config
config_gmm = config.copy()
# Data set up
config_gmm['dataset'] = 'gmm'
config_gmm['input_normalize_sym'] = False
config_gmm['batch_size'] = 100
# Model set up
config_gmm['zdim'] = 1


### MNIST config
config_mnist = config.copy()
# Data set up
config_mnist['dataset'] = 'mnist'
config_mnist['MNIST_data_source_url'] = 'http://yann.lecun.com/exdb/mnist/'
config_mnist['input_normalize_sym'] = False
config_mnist['batch_size'] = 100
# Model set up
config_mnist['zdim'] = 2

### SVHN config
config_svhn = config.copy()
# Data set up
config_svhn['dataset'] = 'svhn'
config_svhn['SVHN_data_source_url'] = 'http://ufldl.stanford.edu/housenumbers/'
config_svhn['input_normalize_sym'] = False
config_svhn['batch_size'] = 100
config_svhn['use_extra'] = False
# Model set up
config_svhn['zdim'] = 16


### CIFAR 10 config
config_cifar10 = config.copy()
# Data set up
config_cifar10['dataset'] = 'cifar10'
config_cifar10['data_dir'] = 'cifar10'
config_cifar10['input_normalize_sym'] = False
config_cifar10['cifar10_data_source_url'] = 'https://www.cs.toronto.edu/~kriz/'
config_cifar10['batch_size'] = 100
# Model set up
config_cifar10['zdim'] = 128


### celebA config
config_celeba = config.copy()
# Data set up
config_celeba['dataset'] = 'celebA'
config_celeba['celeba_data_source_url'] = 'https://docs.google.com/uc?export=download'
config_celeba['celeba_crop'] = 'closecrop' # closecrop, resizecrop
config_celeba['input_normalize_sym'] = False
# config_celeba['batch_size'] = 64
config_celeba['batch_size'] = 100
# Model set up
config_celeba['zdim'] = 64
