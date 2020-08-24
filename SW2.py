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

    # We convert pix positions into pix jumps
    #y_ = torch.cat((torch.zeros(L,B,C,1),y[...,:-1]), dim=-1)
    #y = y - y_
    #y[...,0] = 0


    # Last dim: cumsum weights and resp. pixel pos
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
    #plt.plot(X_icdf[3,0,0,:,0]); plt.plot(X_icdf[3,0,0,:,1].cumsum(-1)); plt.figure()
    X_icdf = torch.cat((X_icdf, torch.ones(L,B,C,N**2,1)), dim=-1)  # (L,B,C,N**2,3)
    Y_icdf = inverse_cdf(projection(Y, L))  # (L,B,C,N**2,2)
    #plt.plot(Y_icdf[3,0,0,:,0]); plt.plot(Y_icdf[3,0,0,:,1].cumsum(-1)); plt.show()
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



    # Ordered time jumps
    diff_p = torch.index_select(concat[...,1].view(-1), -1, indices)  # (L,B,C,2N^2)
    diff_p = diff_p.view(L,B,C,-1)
    plt.plot(diff_p[60,0,0,:]); plt.show()
    p_ = torch.cat((torch.zeros(L,B,C,1),diff_p[...,:-1]), dim=-1)
    diff_p = diff_p - p_

    # Ordered cumsum weights and convert to weights
    diff_w = torch.index_select(concat[...,0].view(-1), -1, indices)  # (L,B,C,2N^2)
    diff_w = diff_w.view(L,B,C,-1)
    w_ = torch.cat((torch.zeros(L,B,C,1),diff_w[...,:-1]), dim=-1)
    diff_w = diff_w - w_

    plus_minus = torch.index_select(concat[...,2].view(-1), -1, indices)
    plus_minus = plus_minus.view(L,B,C,-1)

    #diff_p = torch.cumsum(diff_p, dim=-1)  # (L,B,C,2N^2)

    diff = (diff_w*diff_p*diff_p)  # (L,B,C,2N^2)
    diff = diff.sum(dim=-1)  # (L,B,C)

    return diff, sw








#### testing sw ####
# B=1, C=1, L=4
# SW between X and Y should return 1
X = torch.zeros(1,1,32,32)
X[:,:,0,0] = 1.

Y = torch.zeros(1,1,32,32)
Y[:,:,0,1] = 1.;

diff, sw = sw(X,Y,100)
#sw = sw2(X,Y,4)

print(diff, sw)
plt.plot(diff[:,0,0]); plt.show()

''' There is still a pb'''
