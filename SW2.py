import numpy as np
import torch
import math
import matplotlib.pyplot as plt

def projection(X,L):
    '''This function takes as imput a batch of images, i.e. a tensor of size
    (B,C,N,N), and returns a ....
    '''
    B = X.size(0)
    C = X.size(1)
    N = X.size(-1)

    a = torch.arange(N).repeat(N)  # (N**2)
    b = torch.arange(N).repeat_interleave(N)  # (N**2)
    coord = torch.stack((a, b), dim=0).type(torch.float)  # (2,N**2)

    thetas = torch.arange(L).type(torch.float)/L*np.pi*2  #  (L), may change to random directions, or learn them
    proj = torch.stack((torch.cos(thetas), torch.sin(thetas)), dim=1)  # (L,2)

    coord_proj = torch.matmul(proj, coord)  # (L,N**2)
    coord_proj = coord_proj.unsqueeze(1).unsqueeze(1).repeat(1,B,C,1)  # (L,B,C,N**2)

    X_flat = X.reshape(X.size(0), X.size(1), N**2)  # (B,C,N**2)
    X_flat = X_flat.unsqueeze(0).repeat(L,1,1,1)

    X_proj = torch.stack((coord_proj, X_flat), dim=-1)  # (L,B,C,N**2,2)

    return X_proj


def inverse_cdf(X_proj):
    ''' Takes as input a tensor of size (L,B,C,N**2,2) corresponding to
    sliced images. In X_proj, the last two dims correspond to the positions of
    the Diracs and the respective weights.
    '''

    (L,B,C,N2) = X_proj[...,0].size()
    N = int(math.sqrt(N2))


    # We take the indices of ordered pixels positions
    indices = torch.argsort(X_proj[...,0])  # (L,B,C,N^2)
    indices = indices.view(-1)  # (L*B*C*N^2)
    flat_order = torch.arange(L*B*C).repeat_interleave(N*N)*N*N
    indices = indices+flat_order

    # We sort the weights and take the cum. sum
    x_ = torch.index_select(X_proj[...,1].view(-1), -1, indices)
    x = torch.cumsum(x_.view(L,B,C,N**2), dim=-1)

    # We sort the pixel positions
    y = torch.index_select(X_proj[...,0].view(-1), -1, indices)
    y = y.view(L,B,C,N**2)


    X_p_sorted = torch.stack((x, y), dim=-1)

    return X_p_sorted



def sw (X, Y, L):
    ''' This function takes as input two batches of images, and returns
    the batch sliced wasserstein distance between them. We concatenate
    the icdf of X and Y, take the indexes of the sorted cumsum concatenation
    and ...
    '''
    (B,C,N) = X[...,0].size()

    # At last dim, we add a new col. of +1/-1 for substraction
    X_icdf = inverse_cdf(projection(X, L))  # (L,B,C,N**2,2)
    X_icdf = torch.cat((X_icdf, torch.ones(L,B,C,N**2,1)), dim=-1)  # (L,B,C,N**2,3)
    Y_icdf = inverse_cdf(projection(Y, L)) # (L,B,C,N**2,2)
    Y_icdf = torch.cat((Y_icdf, -torch.ones(L,B,C,N**2,1)), dim=-1) # (L,B,C,N**2,3)

    # We concatenate and take sorting indices
    concat = torch.cat((X_icdf, Y_icdf), dim=-2)  # (L,B,C,2N^2,3)
    indices = torch.argsort(concat[...,0], dim=-1).view(-1)  # (L*B*C*2N^2)
    flat_order = torch.arange(L*B*C).repeat_interleave(2*N*N)*2*N*N
    indices = indices+flat_order


    ##############
    ### benoit ###
    # get ordered cum_sum of X_i (called Ti in overleaf) and add 1 for last Ti
    Ti = torch.index_select(concat[...,1].view(-1), -1, indices)
    Ti = Ti.view(L,B,C,-1)
    Ti = torch.cat((Ti, torch.ones(L,B,C,1)), dim=-1)
    # get order jumps (called wi in overleaf)
    wi = torch.index_select(concat[...,0].view(-1), -1, indices)
    wi = wi.view(L,B,C,-1)
    # get ordered coefs +1/-1
    coef = torch.index_select(concat[...,-1].view(-1), -1, indices)
    coef = coef.view(L,B,C,-1)
    # get square diff of icdf
    diff = torch.cumsum(wi*coef,dim=-1)

    square_diff = (Ti[...,1:] - Ti[...,:-1])  *diff*diff
    # SW
    sw = square_diff.mean(dim=-1).mean(dim=0)
    ##############



    # Ordered pixels
    diff_p = torch.index_select(concat[...,1].view(-1), -1, indices)  # (L,B,C,2N^2)
    diff_p = diff_p.view(L,B,C,-1)
    #p_ = torch.cat((torch.zeros(L,B,C,1),diff_p[...,:-1]), dim=-1)
    #diff_p = diff_p - p_

    # Ordered cumsum weights and convert to weights
    diff_w = torch.index_select(concat[...,0].view(-1), -1, indices)  # (L,B,C,2N^2)
    diff_w = diff_w.view(L,B,C,-1)
    w_ = torch.cat((torch.zeros(L,B,C,1),diff_w[...,:-1]), dim=-1)
    diff_w = diff_w - w_

    plus_minus = torch.index_select(concat[...,2].view(-1), -1, indices)
    plus_minus = plus_minus.view(L,B,C,-1)

    #diff_p = torch.cumsum(diff_p*plus_minus, dim=-1)  # (L,B,C,2N^2)
    #diff_p = diff_p*plus_minus
    plt.plot((diff_w*diff_p*diff_p)[0,0,0,:]); plt.show()

    diff = (diff_w*diff_p*diff_p)  # (L,B,C,2N^2)
    diff = diff.sum(dim=-1)  # (L,B,C)

    return diff, sw



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

    sq_diff = ((pc1_sorted-pc2_sorted)**2).mean(axis=-1).mean(axis=1)

    return sq_diff



def distrib_approx(x, N):
    """
    Wraper to approximate the distribution by a sum od Diracs
    """
    h, w, c = x.get_shape().as_list()[1:]
    batch_size = tf.cast(tf.shape(x)[0], tf.int32)
    L = opts['sw_proj_num']
    # projected image
    sorted_proj, x_sorted = distrib_proj(x)  # (L, h*w), (batch,L,h*w,c)
    # expand sorted_proj for batch and channels
    sorted_proj = sorted_proj.reshape(1,L,1,-1)
    sorted_proj = sorted_proj.repeat(batch_size, axis=0).repeat(c, axis=2) #(batch,L,c,h*w)
    # create the distribution
    x_distrib_val = tf.permute(x_sorted,[0,1,3,2])  #(batch,L,c,h*w)
    dist = Catergorical(probs=x_distrib_val)
    # sample from the distribution N times
    samples = dist.sample(N) # (batch,L,c,N)
    #from the samples, get the pixel values
    point_cloud = tf.gather_nd(sorted_proj, samples, batch_dims=-1)  #(batch,L,c,N)

    return point_cloud


def distrib_proj(x):
    """
    Gets the projected distribution
    """
    h, w, c = x.get_shape().as_list()[1:]
    batch_size = tf.cast(tf.shape(x)[0], tf.int32)
    L = opts['sw_proj_num']
    # get pixel grid projection
    proj = projection(x, L) # (L, h*w)
    # sort proj.
    sorted_proj = tf.sort(proj,axis=-1) # (L, h*w)
    # get proj. argsort
    sorted_indices = tf.argsort(proj,axis=-1) # (L, h*w)
    # create sorted mask
    range = tf.repeat(tf.expand_dims(tf.range(L),axis=-1), N, axis=-1) #(L,N)
    indices = tf.stack([range,sorted_indices], axis=-1) #(L,N,2)
    batch_indices = tf.repeat(tf.expand_dims(indices,axis=0),batch_size,axis=0)
    # sort im. intensities
    x_flat = tf.reshape(x, [-1,1,h*w,c]) # (batch,1,h*w,c)
    x_sorted = tf.gather_nd(tf.repeat(x_flat,L,axis=1), batch_indices, batch_dims=1) #(batch,L,h*w,c)

    return sorted_proj, x_sorted




def projection(x, L):
    """
    Wraper to project images pixels gird into the L diferent directions
    return projections coordinates
    """
    # get coor grid
    h, w, c = x.get_shape().as_list()[1:]
    X,Y = tf.meshgrid(tf.range(h), tf.range(w))
    coord = tf.reshape(tf.stack([X,Y],axis=-1),[-1,2]) # ((h*w)**2,2)
    # get directions to project
    thetas = tf.arange(L) / L *2*m.pi # add other proj methods
    proj_mat = tf.stack([tf.math.cos(thetas),tf.math.sin(thetas)], axis=-1)
    # project grid into proj dir
    proj = tf.linalg.matmul(proj_mat, coord, transpose_b=True) # (L, (h*w)**2)

    return proj




#### testing sw ####
# B=1, C=1, L=4
# SW between X and Y should return 1
X = tf.zeros(1,1,32,32)
X[:,:,0,0] = 1.

Y = torch.tf(1,1,32,32)
Y[:,:,0,1] = 1.;

diff, sw = sw2(X,Y,100)
#sw = sw2(X,Y,4)

#print(diff, sw)
plt.plot(diff[:,0,0]); plt.show()

''' There is still a pb'''
