# %% Import and function declaration
import time
import tensorflow as tf

from source.dl_utils import ae_train_step
from source.utils import create_gif, generate_and_save_images
from source.architectures import AE

SAVE_FOLDER = 'results/000_AE_MNIST/'

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
INPUT_SHAPE = (28, 28, 1)
EPOCHS = 150
ENCODER_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

optimizer = tf.keras.optimizers.Adam(1e-4)
model = AE(INPUT_SHAPE, ENCODER_DIM)

# %% Neural Network - Training
for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    for train_x in train_dataset:
        loss = ae_train_step(model, optimizer, train_x)
    end_time = time.time()

    if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
        print('Epoch: {} | time elapse for current epoch {}'.format(epoch, end_time - start_time))
