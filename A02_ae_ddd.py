# %% Import and function declaration
import cv2
import glob
import datetime
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from source.A01_ae_mnist_utils import AE_Sparse


# Data processing
def process_image(image_path: str) -> np.ndarray:
    """
    Given a path, loads and resizes the image
    :param image_path: image path to process
    :return: image processed
    """
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (540, 258))
    return img


def process_data(data_path: str) -> np.ndarray:
    """
    Given a path, process all the images and packs them as a numpy
    :param data_path: data path
    :return: data processed
    """
    data_filelist = glob.glob(data_path)
    data = np.rollaxis(np.array([process_image(photo) for photo in data_filelist]), 2, 1)
    data = data.astype('float32') / 255.
    return data


# Tensorflow
# @tf.function
# def train_step(image):
#     with tf.GradientTape() as tape:
#         predictions = model(image)
#         loss = loss_object(image, predictions)
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(image, predictions)
#
#     return predictions
#
#
# @tf.function
# def test_step(image):
#     predictions = model(image)
#     t_loss = loss_object(image, predictions)
#
#     test_loss(t_loss)
#     test_accuracy(image, predictions)


# %% Data Load and pre-processing
train_input_filepath = 'data/Denoising Dirty Documents/train/*.png'
train_output_filepath = 'data/Denoising Dirty Documents/train_cleaned/*.png'
test_filepath = 'data/Denoising Dirty Documents/test/*.png'

train_input = process_data(train_input_filepath)
train_output = process_data(train_output_filepath)
test = process_data(test_filepath)

# Dataset Creation
BUFFER = 144
BATCH_SIZE = 32

train_input_dataset = tf.data.Dataset.from_tensor_slices(train_input).shuffle(BUFFER).batch(BATCH_SIZE)
train_output_dataset = tf.data.Dataset.from_tensor_slices(train_output).shuffle(BUFFER).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test).shuffle(BUFFER).batch(BATCH_SIZE)


# %% Neural Network - Definition
INPUT_SHAPE = (540, 258, 1)
EPOCHS = 100
ENCODER_DIM = 128

loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-4)

train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

# TODO: pending of modification
model = AE_Sparse(encoder_dim=ENCODER_DIM)


# Tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_BASE_FOLDER = 'logs/A02_ae_ddd/'

train_log_dir = LOG_BASE_FOLDER + current_time + '/train'
test_log_dir = LOG_BASE_FOLDER + current_time + '/test'
input_image_log_dir = LOG_BASE_FOLDER + current_time + '/input_image'
output_image_log_dir = LOG_BASE_FOLDER + current_time + '/output_image'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
input_image_writer = tf.summary.create_file_writer(input_image_log_dir)
output_image_writer = tf.summary.create_file_writer(output_image_log_dir)
