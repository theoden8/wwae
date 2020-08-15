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
        sliced images. In X_proj, the last two dims correspond to the weights of
        the Diracs and the respective positions.
        '''

        (L,B,C,N2) = X_proj[...,0].size()

        # We build a triangular matrix to transform the weights to the
        # cumulative sum of weights
        cum_sum_mat = torch.flip(torch.triu(torch.ones(N2,N2)), (0,1))  # (N^2,N2)
        cum_sum_mat = cum_sum_mat.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # We take the indices of ordered pixels positions
        X_p_sorted_indices = torch.argsort(X_proj[...,1])  # (L,B,C,N**2)

        # We sort the weights and take the cum. sum
        x_ = torch.index_select(X_proj[...,0], -1, X_p_sorted_indices)
        x = torch.matmul(cum_sum_mat, x_.unsqueeze(-2)).view(L,B,C,N2)

        # We sort the pixel positions
        y = torch.index_select(X_proj[...,1], -1, X_p_sorted_indices)

        # Last dim: ordered pix. positions and respective cumsum of weights
        X_p_sorted = torch.cat((y, x), dim=-1)

        return X_p_sorted


    def sw (X, Y, L):
        ''' This function takes as input two batches of images, and returns
        the batch sliced wasserstein distance between them. We concatenate
        the icdf of X and Y, take the indexes of the sorted cumsum concatenation
        and ...
        '''
        (B,C,N) = X[...,0].size()


        # At last dim, we add a new col. of +1/-1 for substraction
        X_icdf = inverse_cdf(projection(X))  # (L,B,C,N**2,2)
        X_icdf = torch.cat((X_icdf, torch.ones(L,B,C,N**2,1)), dim=-1)
        Y_icdf = inverse_cdf(projection(Y))  # (L,B,C,N**2,2)
        Y_icdf = torch.cat((Y_icdf, -torch.ones(L,B,C,N**2,1)), dim=-1)

        # We concatenate and take sorting indices
        concat = torch.cat((X_icdf, Y_icdf), dim=-2)
        indices = torch.argsort(concat[...,1], dim=-1)

        # Ordered cumsum of weights
        diff0 = torch.index_select(concat[...,1], -1, indices)

        # Ordered times
        diff1 = torch.index_select(concat[...,1], -1, indices)

        # We take the time lapses to reintegrate with coeff +1/-1
        z= torch.cat((torch.zeros(L,B,C,1), diff1[...,:-1]), dim=-1)
        diff1_ = (diff1 - z)*torch.index_select(concat[...,2], -1, indices)

        cum_sum_mat = torch.flip(torch.triu(torch.ones(N2,N2)), (0,1))  # (N^2,N2)
        cum_sum_mat = cum_sum_mat.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        diff1 = torch.matmul(cum_sum_mat, diff1_.unsqueeze(-2)).view(L,B,C,N**2)


        diff = (diff0*diff1*diff1).sum(dim=-1).mean(dim=0)

        return diff
