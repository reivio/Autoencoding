import tensorflow as tf
from time import time
from pathlib import Path
import datetime
from info_vae import *
from data.circle_generator import gen_circles


side_len = 50
train_size = 10e3
test_size = 10e2
batch_size = 32
epochs = 150
print_interval = 1
latent_dim = 3


def make_dataset(generator, size, side_len, batch_size):
    dataset = tf.data.Dataset.from_generator(
        gen_circles,
        args=[size, side_len, side_len//4],
        output_types=(tf.float32),
        output_shapes=((side_len, side_len, 1))
    )
    dataset = dataset.map(lambda x: tf.reshape(x, (1, -1)))
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = make_dataset(gen_circles, train_size, side_len, batch_size)
test_dataset = make_dataset(gen_circles, test_size, side_len, batch_size)

data_point = next(iter(test_dataset))
print(f'\nShape of data element: {data_point.shape}\n')

optimizer = tf.keras.optimizers.Adam(5*10e-4)
model = MMDVAE(side_len, latent_dim)
net_time = 0

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

for epoch in range(epochs):
    start_time = time()
    train_nll = tf.keras.metrics.Mean()
    train_mmd = tf.keras.metrics.Mean()
    for train_x in train_dataset:
        nll, mmd = compute_apply_gradients(model, train_x, optimizer, return_losses=True)
        train_nll(nll)
        train_mmd(mmd)
    end_time = time()
    with train_summary_writer.as_default():
        tf.summary.scalar('NLL', train_nll.result(), step=epoch)
        tf.summary.scalar('MMD', train_mmd.result(), step=epoch)

    if epoch % print_interval == 0:
        test_nll = tf.keras.metrics.Mean()
        test_mmd = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            nll, mmd = compute_loss(model, test_x)
            test_nll(nll)
            test_mmd(mmd)
        with test_summary_writer.as_default():
            tf.summary.scalar('NLL', test_nll.result(), step=epoch)
            tf.summary.scalar('MMD', test_mmd.result(), step=epoch)
        txt = "Epoch: {:>3}. Test set loss: {:7.5f}, NLL: {:7.5f}, MMD: {:7.5f}, time elapsed for current epoch: {:.3f}"
        elapsed_time = end_time - start_time
        net_time += elapsed_time
        print(txt.format(str(epoch), test_nll.result() + test_mmd.result(), test_nll.result(), test_mmd.result(), elapsed_time))
    
    train_nll.reset_states()
    train_mmd.reset_states()
    test_nll.reset_states()
    test_mmd.reset_states()

path = Path('.') / 'saved_models' / 'info_vae' / 'model_weights'
model.save_weights(path)
print(f'Saved model at {path}')
