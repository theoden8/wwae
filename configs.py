import copy
from math import pow, sqrt

### celebA config
config_celebA = {}
# Outputs set up
config_celebA['verbose'] = False
config_celebA['save_every'] = 10000
config_celebA['save_final'] = True
config_celebA['save_train_data'] = True
config_celebA['print_every'] = 100
config_celebA['vizu_embedded'] = False
config_celebA['vizu_sinkhorn'] = True
config_celebA['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_celebA['vizu_encSigma'] = False
config_celebA['vizu_interpolation'] = False
config_celebA['fid'] = False
config_celebA['out_dir'] = 'results_celebA'
config_celebA['plot_num_pics'] = 100
config_celebA['plot_num_cols'] = 10
# Data set up
config_celebA['dataset'] = 'celebA'
config_celebA['celebA_data_source_url'] = 'https://docs.google.com/uc?export=download'
config_celebA['celebA_crop'] = 'closecrop' # closecrop, resizecrop
config_celebA['input_normalize_sym'] = True
# Experiment set up
config_celebA['batch_size'] = 200
config_celebA['epoch_num'] = 101
config_celebA['model'] = 'WAE' #VAE, WAE
config_celebA['use_trained'] = False #train from pre-trained model
# Opt set up
config_celebA['optimizer'] = 'adam' # adam, sgd
config_celebA['adam_beta1'] = 0.9
config_celebA['adam_beta2'] = 0.999
config_celebA['lr'] = 0.001
config_celebA['lr_adv'] = 0.0008
config_celebA['lr_schedule'] = False
config_celebA['normalization'] = 'batchnorm' #batchnorm, layernorm, none
config_celebA['batch_norm_eps'] = 1e-05
config_celebA['batch_norm_momentum'] = 0.99
# Objective set up
config_celebA['cost'] = 'emd' #l2, l2sq, l2sq_norm, l1, emd
config_celebA['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_celebA['sinkhorn_iterations'] = 150 # number of sinkhorn it for emd cost
config_celebA['sinkhorn_reg'] = 1. # regularization param for emd cost
config_celebA['pen_enc_sigma'] = False
config_celebA['lambda_pen_enc_sigma'] = 0.001
# Model set up
config_celebA['zdim'] = 64
config_celebA['pz_scale'] = 1.
config_celebA['prior'] = 'gaussian' # dirichlet, gaussian
config_celebA['encoder'] = 'gauss' # deterministic, gaussian
config_celebA['decoder'] = 'det' # deterministic, gaussian
# lambda set up
config_celebA['beta'] = 10
config_celebA['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_celebA['net_archi'] = 'dcgan'
config_celebA['init_std'] = 0.099999
config_celebA['init_bias'] = 0.0
config_celebA['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_celebA['conv_init'] = 'glorot_uniform' #he, glorot, normilized_glorot, truncated_norm


### MNIST config
config_mnist = {}
# Outputs set up
config_mnist['verbose'] = False
config_mnist['save_every'] = 10000
config_mnist['save_final'] = True
config_mnist['save_train_data'] = True
config_mnist['print_every'] = 100
config_mnist['vizu_embedded'] = False
config_mnist['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_mnist['vizu_encSigma'] = False
config_mnist['vizu_interpolation'] = False
config_mnist['vizu_sinkhorn'] = True
config_mnist['fid'] = False
config_mnist['out_dir'] = 'results_mnist'
config_mnist['plot_num_pics'] = 100
config_mnist['plot_num_cols'] = 10
# Data set up
config_mnist['dataset'] = 'mnist'
config_mnist['MNIST_data_source_url'] = 'http://yann.lecun.com/exdb/mnist/'
config_mnist['input_normalize_sym'] = False

# Experiment set up
config_mnist['batch_size'] = 100
config_mnist['epoch_num'] = 101
config_mnist['model'] = 'WAE' #vae, wae
config_mnist['use_trained'] = False #train from pre-trained model
# Opt set up
config_mnist['optimizer'] = 'adam' # adam, sgd
config_mnist['adam_beta1'] = 0.9
config_mnist['adam_beta2'] = 0.999
config_mnist['lr'] = 0.001
config_mnist['lr_adv'] = 0.0008
config_mnist['lr_schedule'] = False
config_mnist['normalization'] = 'batchnorm' #batchnorm, layernorm, none
config_mnist['batch_norm_eps'] = 1e-05
config_mnist['batch_norm_momentum'] = 0.99
# Objective set up
config_mnist['cost'] = 'emd' #l2, l2sq, l2sq_norm, l1, emd
config_mnist['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_mnist['sinkhorn_iterations'] = 50 # number of sinkhorn it for emd cost
config_mnist['sinkhorn_reg'] = 0.1 # regularization param for emd cost
config_mnist['pen_enc_sigma'] = False
config_mnist['lambda_pen_enc_sigma'] = 0.001
# Model set up
config_mnist['zdim'] = 2
config_mnist['pz_scale'] = 1.
config_mnist['prior'] = 'gaussian' # dirichlet, gaussian
config_mnist['encoder'] = 'gauss' # deterministic, gaussian
config_mnist['decoder'] = 'det' # deterministic, gaussian
# lambda set up
config_mnist['beta'] = 10
config_mnist['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_mnist['net_archi'] = 'mlp'
config_mnist['init_std'] = 0.099999
config_mnist['init_bias'] = 0.0
config_mnist['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_mnist['conv_init'] = 'glorot_uniform' #he, glorot, normilized_glorot, truncated_norm


### SVHN config
config_svhn = {}
# Outputs set up
config_svhn['verbose'] = False
config_svhn['save_every'] = 10000
config_svhn['save_final'] = True
config_svhn['save_train_data'] = True
config_svhn['print_every'] = 100
config_svhn['vizu_embedded'] = False
config_svhn['vizu_sinkhorn'] = True
config_svhn['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_svhn['vizu_encSigma'] = False
config_svhn['vizu_interpolation'] = False
config_svhn['fid'] = False
config_svhn['out_dir'] = 'results_svhn'
config_svhn['plot_num_pics'] = 100
config_svhn['plot_num_cols'] = 10
# Data set up
config_svhn['dataset'] = 'svhn'
config_svhn['SVHN_data_source_url'] = 'http://ufldl.stanford.edu/housenumbers/'
config_svhn['input_normalize_sym'] = False
# Experiment set up
config_svhn['batch_size'] = 128
config_svhn['epoch_num'] = 101
config_svhn['model'] = 'WAE' #VAE, WAE
config_svhn['use_trained'] = False #train from pre-trained model
# Opt set up
config_svhn['optimizer'] = 'adam' # adam, sgd
config_svhn['adam_beta1'] = 0.9
config_svhn['adam_beta2'] = 0.999
config_svhn['lr'] = 0.001
config_svhn['lr_adv'] = 0.0008
config_svhn['lr_schedule'] = False
config_svhn['normalization'] = 'batchnorm' #batchnorm, layernorm, none
config_svhn['batch_norm_eps'] = 1e-05
config_svhn['batch_norm_momentum'] = 0.99
# Objective set up
config_svhn['cost'] = 'emd' #l2, l2sq, l2sq_norm, l1, emd
config_svhn['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_svhn['sinkhorn_iterations'] = 150 # number of sinkhorn it for emd cost
config_svhn['sinkhorn_reg'] = 1. # regularization param for emd cost
config_svhn['pen_enc_sigma'] = False
config_svhn['lambda_pen_enc_sigma'] = 0.001
# Model set up
config_svhn['zdim'] = 8
config_svhn['pz_scale'] = 1.
config_svhn['prior'] = 'gaussian' # dirichlet, gaussian
config_svhn['encoder'] = 'gauss' # deterministic, gaussian
config_svhn['decoder'] = 'det' # deterministic, gaussian
# lambda set up
config_svhn['beta'] = 10
config_svhn['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_svhn['net_archi'] = 'dcgan'
config_svhn['init_std'] = 0.099999
config_svhn['init_bias'] = 0.0
config_svhn['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_svhn['conv_init'] = 'glorot_uniform' #he, glorot, normilized_glorot, truncated_norm

### CIFAR 10 config
config_cifar10 = {}
# Outputs set up
config_cifar10['verbose'] = False
config_cifar10['save_every'] = 10000
config_cifar10['save_final'] = True
config_cifar10['save_train_data'] = True
config_cifar10['print_every'] = 100
config_cifar10['vizu_embedded'] = False
config_cifar10['vizu_sinkhorn'] = True
config_cifar10['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_cifar10['vizu_encSigma'] = False
config_cifar10['vizu_interpolation'] = False
config_cifar10['fid'] = False
config_cifar10['out_dir'] = 'results_svhn'
config_cifar10['plot_num_pics'] = 100
config_cifar10['plot_num_cols'] = 10
# Data set up
config_cifar10['dataset'] = 'cifar10'
config_cifar10['data_dir'] = 'cifar10'
config_cifar10['input_normalize_sym'] = False
config_cifar10['cifar10_data_source_url'] = 'https://www.cs.toronto.edu/~kriz/'
# Experiment set up
config_cifar10['batch_size'] = 128
config_cifar10['epoch_num'] = 101
config_cifar10['model'] = 'WAE' #VAE, WAE
config_cifar10['use_trained'] = False #train from pre-trained model
# Opt set up
config_cifar10['optimizer'] = 'adam' # adam, sgd
config_cifar10['adam_beta1'] = 0.9
config_cifar10['adam_beta2'] = 0.999
config_cifar10['lr'] = 0.001
config_cifar10['lr_adv'] = 0.0008
config_cifar10['lr_schedule'] = False
config_cifar10['normalization'] = 'batchnorm' #batchnorm, layernorm, none
config_cifar10['batch_norm_eps'] = 1e-05
config_cifar10['batch_norm_momentum'] = 0.99
# Objective set up
config_cifar10['cost'] = 'emd' #l2, l2sq, l2sq_norm, l1, emd
config_cifar10['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_cifar10['sinkhorn_iterations'] = 150 # number of sinkhorn it for emd cost
config_cifar10['sinkhorn_reg'] = 1. # regularization param for emd cost
config_cifar10['pen_enc_sigma'] = False
config_cifar10['lambda_pen_enc_sigma'] = 0.001
# Model set up
config_cifar10['zdim'] = 64
config_cifar10['pz_scale'] = 1.
config_cifar10['prior'] = 'gaussian' # dirichlet, gaussian
config_cifar10['encoder'] = 'gauss' # deterministic, gaussian
config_cifar10['decoder'] = 'det' # deterministic, gaussian
# lambda set up
config_cifar10['beta'] = 10
config_cifar10['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_cifar10['net_archi'] = 'dcgan'
config_cifar10['init_std'] = 0.099999
config_cifar10['init_bias'] = 0.0
config_cifar10['mlp_init'] = 'glorot_uniform' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_cifar10['conv_init'] = 'glorot_uniform' #he, glorot, normilized_glorot, truncated_norm
