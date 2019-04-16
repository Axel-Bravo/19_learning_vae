"""
Based on:
- Tensorflow 2.0 Tutorial: https://www.tensorflow.org/alpha/tutorials/generative/cvae
"""

# %% Import and function declaration
import time
import tensorflow as tf

from source.dl_utils import compute_loss, compute_gradients, apply_gradients
from source.utils import create_gif, generate_and_save_images
from source.architectures import CVAE

SAVE_FOLDER = 'results/001_VAE_MNIST/'

# %% Data Load and pre-processing
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

# Normalizing the images to the range of [0., 1.]
train_images /= 255.
test_images /= 255.

# Binarization
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

# Dataset Creation
TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

# %% Neural Network - Definition
EPOCHS = 150
LATENT_DIM = 50
NUM_EXAMPLES_TO_GENERATE = 16

optimizer = tf.keras.optimizers.Adam(1e-4)
random_vector_for_generation = tf.random.normal(shape=[NUM_EXAMPLES_TO_GENERATE, LATENT_DIM])
model = CVAE(LATENT_DIM)

# %% Neural Network - Training
generate_and_save_images(model, 0, random_vector_for_generation, SAVE_FOLDER)

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    for train_x in train_dataset:
        gradients, loss = compute_gradients(model, train_x)
        apply_gradients(optimizer, gradients, model.trainable_variables)
    end_time = time.time()

    if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, ''time elapse for current epoch {}'.format(epoch, elbo,
                                                                                        end_time - start_time))
        generate_and_save_images(model, epoch, random_vector_for_generation, SAVE_FOLDER)

# %% Generate GIF
anim_file = 'cvae.gif'
create_gif(images_folder=SAVE_FOLDER, gif_name=anim_file)
