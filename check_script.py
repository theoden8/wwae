import tensorflow as tf
import logging
import pdb
import numpy as np

logging.error(tf.__version__)

batch_size = 10
zdim = 4
num_pic =3
seed = 123

data = np.arange(batch_size*zdim).reshape((batch_size,zdim))
shuffling_mask = np.arange(batch_size)
np.random.seed(seed)
np.random.shuffle(shuffling_mask)
np.random.seed()
np.random.shuffle(shuffling_mask[num_pic:])
inverse_shuffling_mask = np.argsort(shuffling_mask) #,np.unique(shuffling_mask,return_inverse=True)

# inverse_shuffling_mask[num_pic:] = inverse_shuffling_mask_part + num_pic
# shuffling_mask[num_pic:] = shuffling_mask_part
shuffled_data = data[shuffling_mask]
reconstruct_data = shuffled_data[inverse_shuffling_mask]
idx  = [1,4,6,8,0]
res = shuffled_data[inverse_shuffling_mask[idx]]
test = data[idx]
pdb.set_trace()





reconstruct_data[num_pic:] = reconstruct_data[inverse_shuffling_mask_part+num_pic]
reconstruct_data = reconstruct_data[inverse_shuffling_mask]
# shuffling_mask_part = shuffling_mask[num_pic:]
# inverse_shuffling_mask[num_pic:] = shuffling_mask_part[inverse_shuffling_mask_part]
# reconstruct_data[num_pics:] =  shuffled_data[inverse_shuffling_mask]

# sample_x = tf.constant(np.arange(batch_size*zdim).reshape((batch_size,zdim)),dtype=tf.float32)
# sample_y = tf.constant(np.arange(zdim,(batch_size+1)*zdim).reshape((batch_size,zdim)),dtype=tf.float32)
#
# norms_x = tf.square(sample_x)
# norms_y = tf.square(sample_y)
#
# sample_x = tf.expand_dims(sample_x,-1)
# sample_y = tf.expand_dims(sample_y,-1)
# dotprod = sample_x * tf.transpose(sample_y)
# squared_dist = tf.expand_dims(norms_x,-1) + tf.transpose(norms_y) \
#                 - 2. * dotprod

# shuffle_z = []
# for i in range(z.get_shape()[1]):
#     shuffle_z.append(tf.gather(z[:, i], tf.random.shuffle(tf.range(tf.shape(z[:, i])[0]))))
# shuffled_z = tf.stack(shuffle_z, axis=-1, name="z_shuffled")

# shuffle_mask = [tf.constant(np.random.choice(np.arange(batch_size),batch_size,False)) for i in range(zdim)]
# params = tf.reshape(tf.range(batch_size*zdim,dtype=tf.float32),[batch_size,zdim])
# shuffled_params = []
# for z in np.arange(zdim):
#     shuffled_params.append(tf.gather(params[:,z],shuffle_mask[z],axis=0))
# shuffled_params = tf.stack(shuffled_params,axis=-1)

sess = tf.Session()
x = sess.run(sample_x)[:,:,0]
y = sess.run(sample_y)[:,:,0]
# dotprod = sess.run(dotprod)
# dotprod_ = sess.run(dotprod_)
# for j in range(zdim):
#     for k in range(batch_size):
#         for l in range(batch_size):
#             print('%r'%(x[k,j]*y[l,j]==dotprod[k,j,l]))
# diag_dotprod = sess.run(diag_dotprod)
squared_dist = sess.run(squared_dist)
for j in range(zdim):
    for k in range(batch_size):
        for l in range(batch_size):
            print('%r'%((x[k,j]-y[l,j])**2==squared_dist[k,j,l]))
