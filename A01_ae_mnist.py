# %% Import and function declaration
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from source.A01_ae_mnist_utils import AE_Sparse


@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        predictions = model(image)
        loss = loss_object(image, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(image, predictions)

    return predictions


@tf.function
def test_step(image):
    predictions = model(image)
    t_loss = loss_object(image, predictions)

    test_loss(t_loss)
    test_accuracy(image, predictions)


# %% Data Load and pre-processing
(train_images, _), (test_images, test_images_label) = tf.keras.datasets.mnist.load_data()
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
EPOCHS = 100
ENCODER_DIM = 128

loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-4)

train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

model = AE_Sparse(encoder_dim=ENCODER_DIM)


# Tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_BASE_FOLDER = 'logs/A01_ae_mnist_Sparse/'

train_log_dir = LOG_BASE_FOLDER + current_time + '/train'
test_log_dir = LOG_BASE_FOLDER + current_time + '/test'
input_image_log_dir = LOG_BASE_FOLDER + current_time + '/input_image'
output_image_log_dir = LOG_BASE_FOLDER + current_time + '/output_image'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
input_image_writer = tf.summary.create_file_writer(input_image_log_dir)
output_image_writer = tf.summary.create_file_writer(output_image_log_dir)


#%% Neural Network - Training

# !tensorboard --logdir logs/A01_ae_mnist_Sparse

for epoch in range(EPOCHS):
    image_counter = 0

    for image in train_dataset:
        prediction = train_step(image)

        if image_counter <= 6:  # Limit the number of images per epoch
            with train_summary_writer.as_default():
                tf.summary.image("Input image data", image, max_outputs=6, step=5)
                tf.summary.image("Output image data", prediction, max_outputs=6, step=5)
            image_counter += 1

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for test_image in test_dataset:
        test_step(test_image)

    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    # Console
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))

    # Reset metrics every epoch
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()


#%% Neural Network - Manifold distribution study
x_test_encoded = model.encode(test_images)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=test_images_label)
plt.colorbar()
plt.show()
