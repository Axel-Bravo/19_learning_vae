import numpy as np
import tensorflow as tf

# Convolutional Variational Autoencoder
def cvae_log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def cvae_compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = cvae_log_normal_pdf(z, 0., 0.)
    logqz_x = cvae_log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def cvae_compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = cvae_compute_loss(model, x)
    return tape.gradient(loss, model.trainable_variables), loss


def cvae_apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))
