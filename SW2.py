import numpy as np
import torch

def projection(X,L):
    '''This function takes as imput a batch of images, i.e. a tensor of size
    (B,C,N,N), and returns a ....
    '''
    B = X.size(0)
    C = X.size(1)
    N = X.size(-1)

    a = torch.arange(N).repeat(N)  # (N**2)
    b = torch.arange(N).repeat_interleave(N)  # (N**2)
    coord = torch.stack((a, b), dim=0)  # (2,N**2)

    thetas = torch.arange(L)/L*2*np.pi  #  (L), may change to random directions, or learn them
    proj = torch.stack((torch.cos(thetas), torch.sin(thetas)), dim=1)  # (L,2)

    coord_proj = torch.matmul(proj, coord)  # (L,N**2)
    coord_proj = coord_proj.unsqueeze(1).unsqueeze(1).repeat(1,B,C,1)  # (L,B,C,N**2)

    X_flat = X.reshape(X.size(0), X.size(1), N**2)  # (L,B,C,N**2)
    X_flat = X_flat.unsqueeze(0).repeat(L,1,1,1)

    X_proj = torch.stack((X_flat, coord_proj), dim=-1)  # (L,B,C,N**2,2)

    return X_proj


    def inverse_cdf(X_proj):
        ''' Takes as input a tensor of size (L,B,C,N**2,2) corresponding to
        sliced images. In X_proj, the last two dims correspond to the values of
        the Diracs and the respective positions. For the inverse cdf have to
        sort, permute, and take the cumulative sum.
        '''

        (L,B,C,N2) = X_proj[...0].size()

        cum_sum_mat = torch.flip(torch.triu(torch.ones(N2,N2)), (0,1))  # (N^2,N2)
        cum_sum_mat = cum_sum_mat.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        X_p_sorted_indices = torch.argsort(X_proj[...,1])  # (L,B,C,N**2)

        x_ = torch.index_select(X_proj[...,0], -1, X_p_sorted_indices)
        x = torch.matmul(cum_sum_mat, x_.unsqueeze(-1)).view(L,B,C,N2)
        y = torch.index_select(X_proj[...,1], -1, X_p_sorted_indices)
        X_p_sorted = torch.cat((x,y), dim=-1)

        return X_p_sorted


    def sw (X, Y, L):
        ''' This function takes as input two batches of images, and returns
        the batch sliced wasserstein distance between them
        '''

        X_icdf = inverse_cdf(projection(X))  # (L,B,C,N**2,2)
        Y_icdf = inverse_cdf(projection(Y))  # (L,B,C,N**2,2)
