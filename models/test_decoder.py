import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from vae import CVAE
from data.circle_generator import gen_circles

model = CVAE(3)
model.load_weights('../saved_models/vae/vae_weights')

fig = plt.figure(figsize=(10, 10))


#for index, circle in gen_circles(cols*rows, size):
index = 0

size = 28
cols = 4 
rows = 3
for i in range(rows*cols):
    asd = fig.add_subplot(rows, cols, index+1)
    random_vec = tf.random.uniform(shape=(1, 3))
    random_vec = np.array([[2*i, 10, 10]])
    if i % 2 == 0:
        circle = model.decode(random_vec).numpy()
        asd.title.set_text('Generated')
    else:
        inputs = random_vec
        #inputs = random_vec.numpy()
        print(random_vec, random_vec[0], random_vec[0][0])
        circle = gen_circles(1, size, (inputs[0][0], inputs[0][1]), inputs[0][2])
        circle = next(iter(circle))
        asd.title.set_text('Real')
    plt.imshow(circle.reshape([size, size]))
    print(type(circle))
    #plt.imshow(circle)
    index += 1
plt.show()
