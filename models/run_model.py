import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
from vae import CVAE
#import PIL
#import imageio

from IPython import display

challenge = 'position'



from data.circle_generator import gen_circles
size = 100
train_size = 2000
test_size = 300
batch_size = 500

#gen = gen_circles(train_size, size=size, radius=size//4)

train_dataset = tf.data.Dataset.from_generator(
    gen_circles,
    args=[train_size, size, size//4],
    output_types=(tf.float32),
    output_shapes=((size, size, 1))
).batch(batch_size)

test_dataset = tf.data.Dataset.from_generator(
    gen_circles,
    args=[test_size, size, size//4],
    output_types=(tf.float32),
    output_shapes=((size, size, 1))
).batch(batch_size)


optimizer = tf.keras.optimizers.Adam(5e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

@tf.function
def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


epochs = 75
if challenge == 'position':
  latent_dim = 2
elif challenge == 'with_size':
  latent_dim = 3
model = CVAE(size, latent_dim)


for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    compute_apply_gradients(model, train_x, optimizer)
  end_time = time.time()

  if epoch % 1 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
      loss(compute_loss(model, test_x))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'.format(epoch,
                                                    elbo,
                                                    end_time - start_time))


model.save_weights('../saved_models/vae_l{}_s{}/vae_weights'.format(latent_dim, size))
