#%% Import and function declaration
import cv2
import glob
import datetime
import numpy as np
import tensorflow as tf
from source.A02_ae_ddd_utils import AE


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


@tf.function
def train_step(image_input, image_output):
    with tf.GradientTape() as tape:
        predictions = model(image_input)
        loss = loss_object(image_output, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(image_output, predictions)

    return predictions


#%% Data Load and pre-processing
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


#%% Neural Network - Definition
INPUT_SHAPE = (540, 258, 1)
EPOCHS = 100
ENCODER_DIM = 128

loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-4)

train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

model = AE(encoder_dim=ENCODER_DIM)


# Tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_BASE_FOLDER = 'logs/A02_ae_ddd/'

train_log_dir = LOG_BASE_FOLDER + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


#%% Neural Network - Training

# !tensorboard --logdir logs/A02_ae_ddd

for epoch in range(EPOCHS):
    image_counter = 0

    for i_image_input, i_image_output in train_input_dataset, train_output_dataset:
        prediction = train_step(image_input=i_image_input, image_output=i_image_output)

        if image_counter <= 6:  # Limit the number of images per epoch
            with train_summary_writer.as_default():
                tf.summary.image("Input image data", i_image_input, max_outputs=6, step=5)
                tf.summary.image("Output image data", i_image_output, max_outputs=6, step=5)
                tf.summary.image("Predicted image data", prediction, max_outputs=6, step=5)
            image_counter += 1

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    # Console
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100))

    # Reset metrics every epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
