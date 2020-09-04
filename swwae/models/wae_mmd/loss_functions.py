import tensorflow as tf
import keras.backend as K


def total_loss(opts, sample_qz, batch_size, mmd_weight, recon_loss_func=None):
    def _total_loss(y_true, y_pred):
        if recon_loss_func is None:
            recon_loass = per_pix_recon_loss(y_true, y_pred)
        else:
            recon_loass = recon_loss_func(y_true, y_pred)
        mmd_div = mmd_loss(sample_qz, batch_size, opts)()
        return recon_loass + mmd_weight*mmd_div

    return _total_loss


def per_pix_recon_loss(y_true, y_pred):
    # Coppied from https://github.com/tolstikhin/wae/blob/63515656201eb6e3c3f32f6d38267401ed8ade8f/wae.py
    # opts['cost'] == 'l2sq'
    # c(x,y) = ||x - y||_2^2
    loss = tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2, 3])
    loss = 0.05 * tf.reduce_mean(loss)
    return loss


def mmd_loss(sample_qz, batch_size, opts):
    def _mmd_penalty(y_true=0, y_pred=0):
        '''Coppied from https://github.com/tolstikhin/wae/blob/master/wae.py'''
        sample_pz = K.random_normal(shape=(batch_size, opts['zdim']), mean=0.0, stddev=opts['pz_scale'])
        sigma2_p = opts['pz_scale'] ** 2
        kernel = opts['mmd_kernel']
        n = batch_size
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = (n * n - n) / 2

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        # if opts['verbose']:
        #     distances = tf.Print(
        #         distances,
        #         [tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]],
        #         'Maximal Qz squared pairwise distance:')
        #     distances = tf.Print(distances, [tf.reduce_mean(distances_qz)],
        #                         'Average Qz squared pairwise distance:')

        #     distances = tf.Print(
        #         distances,
        #         [tf.nn.top_k(tf.reshape(distances_pz, [-1]), 1).values[0]],
        #         'Maximal Pz squared pairwise distance:')
        #     distances = tf.Print(distances, [tf.reduce_mean(distances_pz)],
        #                         'Average Pz squared pairwise distance:')

        if kernel == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            # Maximal heuristic for the sigma^2 of Gaussian kernel
            # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
            # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
            # sigma2_k = opts['latent_space_dim'] * sigma2_p
            if opts['verbose']:
                sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
            res1 = tf.exp(- distances_qz / 2. / sigma2_k)
            res1 += tf.exp(- distances_pz / 2. / sigma2_k)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = tf.exp(- distances / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':
            # k(x, y) = C / (C + ||x - y||^2)
            # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            if opts['pz'] == 'normal':
                Cbase = 2. * opts['zdim'] * sigma2_p
            elif opts['pz'] == 'sphere':
                Cbase = 2.
            elif opts['pz'] == 'uniform':
                # E ||x - y||^2 = E[sum (xi - yi)^2]
                #               = zdim E[(xi - yi)^2]
                #               = const * zdim
                Cbase = opts['zdim']
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2

        return stat
        # return K.abs(stat)
    return _mmd_penalty


def sw2(opts, x1, x2):
    """
    Compute the sliced-wasserstein distance of x1 and x2
    in the pixel space
    x1,2: [batch_size, height, width, channels]
    """
    h, w, c = x.get_shape().as_list()[1:]
    N = opts['sw_samples_num']
    # get distributions approx.
    pc1 = distrib_approx(x1, N)
    pc2 = distrib_approx(x2, N)
    # sort the point clouds
    pc1_sorted = tf.sort(pc1, axis=-1)  # (batch,L,c,N)
    pc2_sorted = tf.sort(pc2, axis=-1)  # (batch,L,c,N)

    sq_diff = tf.math.reduce_mean((pc1_sorted-pc2_sorted)**2, axis=-1)  # (batch,L,c)
    sq_diff = tf.math.reduce_mean(sq_diff, axis=1)  # (batch,c)

    return sq_diff



def distrib_approx(x, N):
    """
    Wraper to approximate the distribution by a sum od Diracs
    """
    h, w, c = x.get_shape().as_list()[1:]
    B = tf.cast(tf.shape(x)[0], tf.int32)
    L = opts['sw_proj_num']
    # projected image
    sorted_proj, x_sorted = distrib_proj(x)  # (L, h*w), (batch,L,c,h*w)
    # expand sorted_proj for batch and channels
    sorted_proj = tf.reshape(sorted_proj,[1,L,1,-1])
    sorted_proj = tf.tile(sorted_proj, [B,1,c,1]) #(batch,L,c,h*w)
    # create the distribution
    dist = tfp.distributions.Categorical(probs=x_sorted)
    # sample from the distribution N times
    samples = dist.sample(N) # (N,batch,L,c)
    samples = tf.transpose(samples, [1,2,3,0])

    i_b = tf.tile(tf.reshape(tf.range(B), [B,1,1,1]), [1,L,c,N])
    i_L = tf.tile(tf.reshape(tf.range(L), [1,L,1,1]), [B,1,c,N])
    i_c = tf.tile(tf.reshape(tf.range(c), [1,1,c,1]), [B,L,1,N])

    indices = tf.stack([i_b,i_L,i_c,samples], axis=-1)
    #from the samples, get the pixel values
    point_cloud = tf.gather_nd(sorted_proj, indices)  #(batch,L,c,N)

    return point_cloud


def distrib_proj(x):
    """
    Gets the projected distribution
    """
    h, w, c = x.get_shape().as_list()[1:]
    B = tf.cast(tf.shape(x)[0], tf.int32)
    L = opts['sw_proj_num']
    # get pixel grid projection
    proj = projection(x,L) # (L, h*w)
    # sort proj.
    sorted_proj = tf.sort(proj,axis=-1) # (L, h*w)
    # get proj. argsort
    sorted_indices = tf.argsort(proj,axis=-1) # (L, h*w)
    # create sort indices
    b_indices = tf.tile(tf.expand_dims(sorted_indices,axis=0),[B,1,1]) # (B,L,h*w)
    bc_indices = tf.tile(tf.expand_dims(b_indices,axis=2),[1,1,c,1]) # (B,L,c,h*w)

    i_b = tf.tile(tf.reshape(tf.range(B), [B,1,1,1]), [1,L,c,h*w])
    i_L = tf.tile(tf.reshape(tf.range(L), [1,L,1,1]), [B,1,c,h*w])
    i_c = tf.tile(tf.reshape(tf.range(c), [1,1,c,1]), [B,L,1,h*w])

    indices = tf.stack([i_b,i_L,i_c,bc_indices], axis=-1)

    # sort im. intensities
    x_flat = tf.transpose(tf.tile(tf.reshape(x, [-1,1,h*w,c]),[1,L,1,1]),[0,1,3,2]) # (batch,L,c,h*w)
    x_sorted = tf.gather_nd(x_flat, indices) #(batch,L,c,h*w)

    return sorted_proj, x_sorted


def projection(x,L):
    """
    Wraper to project images pixels gird into the L diferent directions
    return projections coordinates
    """
    # get coor grid
    h, w, c = x.get_shape().as_list()[1:]
    X,Y = tf.meshgrid(tf.range(h), tf.range(w))
    coord = tf.cast(tf.reshape(tf.stack([X,Y],axis=-1),[-1,2]),tf.float32) # ((h*w),2)
    # get directions to project
    thetas = tf.range(L, dtype=tf.float32) / L *2*np.pi # add other proj methods
    proj_mat = tf.stack([tf.math.cos(thetas),tf.math.sin(thetas)], axis=-1)
    # project grid into proj dir
    proj = tf.linalg.matmul(proj_mat, coord, transpose_b=True) # (L, (h*w))

    return proj

