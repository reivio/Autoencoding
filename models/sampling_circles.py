import tensorflow as tf
from matplotlib import pyplot as plt
from data.circle_generator import gen_circles

model_path = './saved_models/vae/vae_weights'
latent_dim = 3
num_examples_to_generate = 6

class VAE(tf.keras.Model):
  def __init__(self, latent_dim):
    super(VAE, self).__init__()
    self.latent_dim = latent_dim
    self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
          tf.keras.layers.Conv2D(
              filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Conv2D(
              filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Flatten(),
          # No activation
          tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )

    self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
          tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
          tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
          tf.keras.layers.Conv2DTranspose(
              filters=64,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          tf.keras.layers.Conv2DTranspose(
              filters=32,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          # No activation
          tf.keras.layers.Conv2DTranspose(
              filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits

model = VAE(latent_dim)
model.load_weights(model_path)
print("\n\nsdfg\ngaer\nergtfgar\n\njhlk\n\nhgkhj")

random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
print("rand vec: {}".format(random_vector_for_generation))

predictions = model.sample(random_vector_for_generation)
fig = plt.figure(figsize=(4, 4))

for i in range(predictions.shape[0]):
  plt.subplot(4, 4, i+1)
  plt.imshow(predictions[i, :, :, 0], cmap='gray')
  plt.axis('off')

size = 28

circles = []
for vec in random_vector_for_generation:
    loc = vec[0], vec[1]
    rad = vec[2]
    circles.append(gen_circles(1, size=size, location=loc, radius=rad))



if __name__ == '__main__':

    fig = plt.figure(figsize=(10, 10))
    cols = num_examples_to_generate
    rows = 2

    for index, pred in enumerate(predictions):
        print("img added")
        fig.add_subplot(rows, cols, index+1)
        plt.imshow(pred.reshape([size, size]))

        circle = circles[index]
        fig.add_subplot(rows, cols, 2*(index+1))
        plt.imshow(circle.reshape([size, size]))
    plt.show()
