import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Dropout, Reshape
from tensorflow.keras import Model


# 1 Spare Autoencoder
class AE_Sparse(Model):
    def __init__(self, encoder_dim):
        """
        Convolutional Autoencoder
        """
        super(AE_Sparse, self).__init__()
        self.encoder_dim = encoder_dim
        self.regularizer = tf.keras.regularizers.l1(l=0.05)
        self.drop = Dropout(rate=0.3)

        # Encoder
        self.conv_1 = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu',
                             kernel_regularizer=self.regularizer)
        self.conv_2 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu',
                             kernel_regularizer=self.regularizer)
        self.flatten = Flatten()
        self.dense_1 = Dense(self.encoder_dim)

        # Decoder
        self.dense_1_dec = Dense(units=7 * 7 * 32, activation=tf.nn.relu)
        self.reshape = Reshape(target_shape=(7, 7, 32))
        self.conv_tp_1 = Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu',
                                         kernel_regularizer=self.regularizer)
        self.conv_tp_2 = Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu',
                                         kernel_regularizer=self.regularizer)
        self.conv_tp_3 = Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME",
                                         kernel_regularizer=self.regularizer)

    def encode(self, x):
        x = self.conv_1(x)
        x = self.drop(x)
        x = self.conv_2(x)
        x = self.drop(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        return x

    def decode(self, x):
        x = self.dense_1_dec(x)
        x = self.reshape(x)
        x = self.drop(x)
        x = self.conv_tp_1(x)
        x = self.drop(x)
        x = self.conv_tp_2(x)
        x = self.drop(x)
        x = self.conv_tp_3(x)
        return x

    def call(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


# 1 Base
class AE(Model):
    def __init__(self, encoder_dim):
        """
        Convolutional Autoencoder
        """
        super(AE, self).__init__()
        self.encoder_dim = encoder_dim

        # Encoder
        self.conv_1 = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')
        self.conv_2 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')
        self.flatten = Flatten()
        self.dense_1 = Dense(self.encoder_dim)

        # Decoder
        self.dense_1_dec = Dense(units=7 * 7 * 32, activation=tf.nn.relu)
        self.reshape = Reshape(target_shape=(7, 7, 32))
        self.conv_tp_1 = Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')
        self.conv_tp_2 = Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')
        self.conv_tp_3 = Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME")

    def encode(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        return x

    def decode(self, x):
        x = self.dense_1_dec(x)
        x = self.reshape(x)
        x = self.conv_tp_1(x)
        x = self.conv_tp_2(x)
        x = self.conv_tp_3(x)
        return x

    def call(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x