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
config_celebA['batch_size'] = 128
config_celebA['epoch_num'] = 101
config_celebA['model'] = 'WAE' #VAE, WAE
config_celebA['use_trained'] = False #train from pre-trained model
# Opt set up
config_celebA['optimizer'] = 'adam' # adam, sgd
config_celebA['adam_beta1'] = 0.9
config_celebA['adam_beta2'] = 0.999
config_celebA['lr'] = 0.0001
config_celebA['lr_adv'] = 0.0008
config_celebA['lr_schedule'] = False
config_celebA['normalization'] = 'batchnorm' #batchnorm, layernorm, none
config_celebA['batch_norm_eps'] = 1e-05
config_celebA['batch_norm_momentum'] = 0.99
config_celebA['dropout_rate'] = 1.
# Objective set up
config_celebA['cost'] = 'emd' #l2, l2sq, l2sq_norm, l1, emd
config_celebA['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_celebA['sinkhorn_iterations'] = 50 # number of sinkhorn it for emd cost
config_celebA['sinkhorn_reg'] = 1. # regularization param for emd cost
config_celebA['pen_enc_sigma'] = False
config_celebA['lambda_pen_enc_sigma'] = 0.001
# Model set up
config_celebA['zdim'] = 10
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
config_mnist['batch_size'] = 128
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
config_mnist['dropout_rate'] = 1.
# Objective set up
config_mnist['cost'] = 'emd' #l2, l2sq, l2sq_norm, l1, emd
config_mnist['mmd_kernel'] = 'IMQ' # RBF, IMQ
config_mnist['sinkhorn_iterations'] = 100 # number of sinkhorn it for emd cost
config_mnist['sinkhorn_reg'] = 0.1 # regularization param for emd cost
config_mnist['pen_enc_sigma'] = False
config_mnist['lambda_pen_enc_sigma'] = 0.001
# Model set up
config_mnist['zdim'] = 10
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
config_svhn['save_every'] = 2000
config_svhn['print_every'] = 200000
config_svhn['save_final'] = True
config_svhn['save_train_data'] = False
config_svhn['vizu_sinkhorn'] = False
config_svhn['vizu_embedded'] = True
config_svhn['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_svhn['vizu_encSigma'] = False
config_svhn['fid'] = False
config_svhn['out_dir'] = 'results_svhn'
config_svhn['plot_num_pics'] = 100
config_svhn['plot_num_cols'] = 10
# Data set up
config_svhn['dataset'] = 'svhn'
config_svhn['SVHN_data_source_url'] = 'http://ufldl.stanford.edu/housenumbers/'
config_svhn['input_normalize_sym'] = False
# Experiment set up
config_svhn['train_dataset_size'] = -1
config_svhn['use_extra'] = False
config_svhn['batch_size'] = 128
config_svhn['epoch_num'] = 4120
config_svhn['model'] = 'wae' #vae, wae
config_svhn['use_trained'] = False #train from pre-trained model
# Opt set up
config_svhn['optimizer'] = 'adam' # adam, sgd
config_svhn['adam_beta1'] = 0.5
config_svhn['lr'] = 0.0002
config_svhn['lr_adv'] = 0.0008
config_svhn['lr_schedule'] = False
config_svhn['normalization'] = 'None' #batchnorm, layernorm, none
config_svhn['batch_norm_eps'] = 1e-05
config_svhn['batch_norm_momentum'] = 0.99
config_svhn['dropout_rate'] = 1.
# Objective set up
config_svhn['coef_rec'] = 1. # coef recon loss
config_svhn['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_svhn['mmd_kernel'] = 'IMQ' # RBF, IMQ
# Model set up
config_svhn['nlatents'] = 8
config_svhn['zdim'] = [64,49,36,25,16,9,4,2]
config_svhn['pz_scale'] = 1.
config_svhn['prior'] = 'gaussian' # dirichlet or gaussian
config_svhn['encoder'] = ['gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_svhn['decoder'] = ['det','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
# lambda set up
config_svhn['lambda_scalar'] = 10.
config_svhn['lambda'] = [1/config_svhn['zdim'][i] for i in range(config_svhn['nlatents'])]
config_svhn['lambda'].append(0.0001/config_svhn['zdim'][-1])
config_svhn['lambda_schedule'] = 'constant' # adaptive, constant
# NN set up
config_svhn['init_std'] = 0.099999
config_svhn['init_bias'] = 0.0
config_svhn['mlp_init'] = 'glorot_he' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_svhn['conv_init'] = 'glorot_uniform' #he, glorot, normilized_glorot, truncated_norm
config_svhn['e_nlatents'] = config_svhn['nlatents'] #config_mnist['nlatents']

### CIFAR 10 config
config_cifar10 = {}
# Outputs set up
config_cifar10['verbose'] = False
config_cifar10['save_every'] = 2000
config_cifar10['print_every'] = 200000
config_cifar10['save_final'] = True
config_cifar10['save_train_data'] = False
config_cifar10['vizu_sinkhorn'] = False
config_cifar10['vizu_embedded'] = True
config_cifar10['embedding'] = 'umap' #vizualisation method of the embeddings: pca, umap
config_cifar10['vizu_encSigma'] = False
config_cifar10['fid'] = False
config_cifar10['work_dir'] = 'results_cifar'
config_cifar10['plot_num_pics'] = 100
config_cifar10['plot_num_cols'] = 10

# Data set up
config_cifar10['dataset'] = 'cifar10'
config_cifar10['data_dir'] = 'cifar10'
config_cifar10['input_normalize_sym'] = False
config_cifar10['cifar10_data_source_url'] = 'https://www.cs.toronto.edu/~kriz/'

# Experiment set up
config_cifar10['train_dataset_size'] = -1
config_cifar10['batch_size'] = 128
config_cifar10['epoch_num'] = 4120
config_cifar10['model'] = 'wae' #vae, wae
config_cifar10['use_trained'] = False #train from pre-trained model
config_cifar10['e_pretrain'] = False #pretrained the encoder parameters
config_cifar10['e_pretrain_sample_size'] = 200
config_cifar10['e_pretrain_it'] = 1000

# Opt set up
config_cifar10['optimizer'] = 'adam' # adam, sgd
config_cifar10['adam_beta1'] = 0.5
config_cifar10['lr'] = 0.0001
config_cifar10['lr_adv'] = 0.0008
config_cifar10['e_norm'] = 'batchnorm' #batchnorm, layernorm, none
config_cifar10['d_norm'] = 'layernorm' #batchnorm, layernorm, none
config_cifar10['batch_norm_eps'] = 1e-05
config_cifar10['batch_norm_momentum'] = 0.99

# Objective set up
config_cifar10['coef_rec'] = 1. # coef recon loss
config_cifar10['cost'] = 'l2sq' #l2, l2sq, l2sq_norm, l1
config_cifar10['penalty'] = 'mmd' #sinkhorn, mmd
config_cifar10['pen'] = 'wae' #wae, wae_mmd
config_cifar10['epsilon'] = 0.1 #Sinkhorn regularization parameters
config_cifar10['L'] = 30 #Sinkhorn iteration
config_cifar10['mmd_kernel'] = 'RQ' # RBF, IMQ, RQ

# Model set up
config_cifar10['nlatents'] = 8
config_cifar10['zdim'] = [64,49,36,25,16,9,4,2]
config_cifar10['pz_scale'] = 1.
config_cifar10['prior'] = 'gaussian' # dirichlet or gaussian

# lambda set up
config_cifar10['lambda_scalar'] = 10.
config_cifar10['lambda'] = [1/config_cifar10['zdim'][i] for i in range(config_cifar10['nlatents'])]
config_cifar10['lambda'].append(0.0001/config_cifar10['zdim'][-1])
config_cifar10['lambda_schedule'] = 'constant' # adaptive, constant

# NN set up
config_cifar10['init_std'] = 0.0099999
config_cifar10['init_bias'] = 0.0
config_cifar10['mlp_init'] = 'glorot_he' #normal, he, glorot, glorot_he, glorot_uniform, ('uniform', range)
config_cifar10['conv_init'] = 'he' #he, glorot, normilized_glorot, truncated_norm
config_cifar10['filter_size'] = [5,3,3,3,3,3,3,3]
config_cifar10['last_archi'] = ['conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','conv1x1','dense']


config_cifar10['e_nlatents'] = config_cifar10['nlatents'] #config_mnist['nlatents']
config_cifar10['encoder'] = ['gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_cifar10['e_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, ali, began
config_cifar10['e_nlayers'] = [2,2,2,2,2,2,2,2]
config_cifar10['e_nfilters'] = [96,96,64,64,32,32,32,32]
config_cifar10['e_nonlinearity'] = 'leaky_relu' # soft_plus, relu, leaky_relu, tanh


config_cifar10['decoder'] = ['det','gauss','gauss','gauss','gauss','gauss','gauss','gauss'] # deterministic, gaussian
config_cifar10['d_arch'] = ['dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan','dcgan'] # mlp, dcgan, dcgan_mod, ali, began
config_cifar10['d_nlayers'] = [2,2,2,2,2,2,2,2]
config_cifar10['d_nfilters'] = [96,96,64,64,32,32,32,32]
config_cifar10['d_nonlinearity'] = 'relu' # soft_plus, relu, leaky_relu, tanh
