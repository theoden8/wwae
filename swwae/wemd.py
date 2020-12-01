import tensorflow as tf
import numpy as np
import scipy.io as sio



'''
def wemd(x1,x2):

    h, w, c = x1.get_shape().as_list()[1:]

    mass1 = tf.reduce_sum(x1, axis=[1,2]) #[b,c]
    xs1 = x1 / tf.reshape(mass1, [-1,1,1,c])
    mass2 = tf.reduce_sum(x2, axis=[1,2]) #[b,c]
    xs2 = x2 / tf.reshape(mass2, [-1,1,1,c])


    d = xs1 - xs2
    d = tf.cast(d, tf.complex64)
    d = tf.transpose(d, [0,3,1,2])  # (b,c,h,w)

    matfilters = sio.loadmat('./matlab/filters/bumpsteerableg1_fft2d_N64_J6_L8.mat')
    J = 6
    L = 16
    fftphi = matfilters['filt_fftphi'].astype(np.complex_)
    fftpsi = matfilters['filt_fftpsi'].astype(np.complex_)  # (J,L,h,w)

    fftd = tf.signal.fft2d(d)
    fftd = tf.reshape(fftd, [-1,c,1,1,h,w])

    fftpsi = tf.cast(tf.reshape(fftpsi, [1,1,J,L,h,w]), tf.complex64)

    hatprod = fftpsi*fftd

    prod = tf.signal.ifft2d(hatprod)

    mod = tf.math.abs(prod)  # (b,c,J,L,h,w)
    mod_mean = tf.reduce_mean(mod, axis=[-5,-3,-2,-1])

    coef = [2**(j/2) for j in [4,3,2,1,0]]
    coef = tf.reshape(coef, [1,5])
    coef = tf.cast(coef, tf.float32)

    wemd = tf.reduce_mean(1*mod_mean, axis=-1)

    diff_m = (1. - mass2/mass1)**2
    diff_m = tf.math.reduce_mean(diff_m, axis=-1)

#    print(diff_m.shape, wemd.shape)

    return wemd #+ 1e-12*diff_m
'''

def wemd(opts, waves, x1, x2):

    h, w, c = x1.get_shape().as_list()[1:]

    J = int(np.log2(h))  # number of scales
    L = 8  # number of orientations

    mass1 = tf.reduce_sum(x1, axis=[1,2]) #[b,c]
    xs1 = x1 / tf.reshape(mass1, [-1,1,1,c])
    mass2 = tf.reduce_sum(x2, axis=[1,2]) #[b,c]
    xs2 = x2 / tf.reshape(mass2, [-1,1,1,c])


    d = xs1 - xs2
    d = tf.cast(d, tf.complex64)
    d = tf.transpose(d, [0,3,1,2])  # (b,c,h,w)

    fftd = tf.signal.fft2d(d)
    fftd = tf.reshape(fftd, [-1,c,1,1,h,w])  # (b,c,1,1,h,w)

    waves = tf.reshape(waves, [1,1,J,L,h,w])  # (1,1,J,L,h,w)

    hatprod = waves*fftd  # (b,c,J,L,h,w)

    prod = tf.signal.ifft2d(hatprod)

    mod = tf.math.abs(prod)  # (b,c,J,L,h,w)

    mod_mean = tf.reduce_mean(mod, axis=[-5,-3,-2,-1])  # (b,J)

    # coef for putting different wieghts to different scales as in the paper
    coef = [2**(j/2) for j in [4,3,2,1,0]]
    coef = tf.reshape(coef, [1,5])
    coef = tf.cast(coef, tf.float32)

    wemd = tf.reduce_mean(1*mod_mean, axis=-1)  # (b,)

    diff_m = (1. - mass2/mass1)**2
    diff_m = tf.math.reduce_mean(diff_m, axis=-1)


    return wemd #+ opts['gamma']*diff_m  # gamma should be very low, as wavelet moments have small norm
