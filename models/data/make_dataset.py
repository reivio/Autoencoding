import tensorflow as tf
from circle_generator import gen_circles

size = 28
dataset = tf.data.Dataset.from_generator(
    gen_circles,
    args=[3, size],
    output_types=(tf.float32),
    output_shapes=((size, size, 1))
)

for img in dataset.batch(2):
    print(img.shape)
    print(img)

