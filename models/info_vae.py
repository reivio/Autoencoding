import tensorflow as tf
import numpy as np
from pathlib import Path


class MMDVAE(tf.keras.Model):
    """Variational Autoencoder
    Kullback-Leibler divergence replaced with Maximum Mean Discrepancy
    in order to increase the informativeness of the latent space
    """
    def __init__(self, side_len, latent_dim):
        super(MMDVAE, self).__init__()
        self.latent_dim = latent_dim
        #hidden_dim = (side_len ** 2) // 7
        hidden_dim = 30
        self.inference_net = tf.keras.Sequential(
            [   
                tf.keras.layers.Dense(side_len ** 2, activation='relu'),
                tf.keras.layers.Dense(hidden_dim, activation='relu'),
                tf.keras.layers.Dense(latent_dim)
            ]
        )
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(latent_dim, activation='relu'),
                tf.keras.layers.Dense(hidden_dim, activation='relu'),
                tf.keras.layers.Dense(side_len ** 2, activation='relu')
            ]
        )

    # @tf.function
    # def sample(self, eps=None):
    #     if eps == None:
    #         eps = tf.random.normal(shape=(100, self.latent_dim))
    #     return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        z = self.inference_net(x)
        return z

    # def reparameterize(self, mean, logvar):
    #     eps = tf.random.normal(shape=mean.shape)
    #     return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

@tf.function
def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

@tf.function
def compute_mmd(x, y, sigmasqr=1.0):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

@tf.function
def compute_loss(model, x):
    z = model.encode(x)
    x_hat = model.decode(z)
    true_samples = tf.random.normal(z.shape.as_list())

    true_samples = tf.squeeze(true_samples)
    z = tf.squeeze(z)

    loss_mmd = compute_mmd(true_samples, z)
    loss_nll = tf.reduce_mean(tf.square(x_hat - x))
    return loss_nll, loss_mmd


@tf.function
def compute_apply_gradients(model, x, optimizer, beta=1, return_losses=False):
    with tf.GradientTape() as tape:
        loss_nll, loss_mmd = compute_loss(model, x)
        loss = loss_nll + beta * loss_mmd
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if return_losses:
        return loss_nll, loss_mmd
