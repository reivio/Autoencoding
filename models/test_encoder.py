import tensorflow as tf
import numpy as np

from vae import CVAE
from data.circle_generator import gen_circles

model = CVAE(2)
model.load_weights('../saved_models/vae/vae_weights')

numTests = 10
size = 28

dataset = tf.data.Dataset.from_generator(
    gen_circles,
    args=[3, size],
    output_types=(tf.float32),
    output_shapes=((size, size, 1))
)

for img in dataset:
    