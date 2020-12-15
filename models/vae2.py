import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data.circle_generator import gen_circles

size = 28
dataset_size = 10000
batch_size = 256

train_dataset = tf.data.Dataset.from_generator(
    gen_circles,
    args=[dataset_size, size],
    output_types=(tf.float32),
    output_shapes=((size, size, 1))
).batch(batch_size)

test_dataset = tf.data.Dataset.from_generator(
    gen_circles,
    args=[dataset_size, size],
    output_types=(tf.float32),
    output_shapes=((size, size, 1))
).batch(batch_size)

encoder_input = keras.Input(shape=(28, 28, 1), name='img')
x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(16, 3, activation='relu')(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()

x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)

autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
autoencoder.summary()
autoencoder.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanSquaredError()]
)

autoencoder.fit(
    train_dataset.map(lambda img: (img, img)),
    epochs=10
)
