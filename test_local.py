import numpy as np
import tensorflow as tf

import pdb

bnum = 5
N, L, c = 4, 3, 1

x = np.arange(bnum*N*1*c).reshape([bnum, 1, N, c]) #(bnum,1,N,c)
idx = np.stack([np.arange(N) for _ in range(L)], axis=0) #(L,N)
for l in range(L):
    idx[l] = np.random.permutation(idx[l])

sess = tf.InteractiveSession()
x_tf = tf.constant(x,dtype=tf.int32)
idx_tf = tf.constant(idx,dtype=tf.int32)

range = tf.repeat(tf.expand_dims(tf.range(L),axis=-1), N, axis=-1) #(L,N)
indices = tf.stack([range,idx_tf], axis=-1) #(L,N,2)
batch_indices = tf.repeat(tf.expand_dims(indices,axis=0),bnum,axis=0)
# x_1d = tf.gather_nd(tf.repeat(x_tf[0],L,axis=0), indices)
pdb.set_trace()
x_batch = tf.gather_nd(tf.repeat(x_tf,L,axis=1), batch_indices, batch_dims=1)
print(x_batch[0].eval())

# x_rep = tf.repeat(x_tf, L, axis=1) #(bnum,L,N,c)
# range_rep = tf.reshape(tf.repeat(tf.range(N,dtype=tf.int32), L*N, axis=0),(L,N,N))
# idx_rep = tf.repeat(tf.expand_dims(idx_tf,axis=0),bnum, axis=0)
#
#
# idx_rep = tf.repeat(tf.stack([idx_tf,range_rep],axis=-1), bnum, axis=0) #(bnum,L,N)
#
#
# x_sort = tf.gather_nd(x_rep, idx_rep, batch_dims=0)
pdb.set_trace()
