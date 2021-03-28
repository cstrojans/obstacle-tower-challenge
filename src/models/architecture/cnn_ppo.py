import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
import math


class CNN(keras.Model):
    def __init__(self, action_size, ip_shape=(84, 84, 3)):
        super(CNN, self).__init__()
        self.action_size = action_size
        self.ip_shape = ip_shape

        self.conv1 = layers.Conv2D(filters=32,
                                   kernel_size=(8, 8),
                                   strides=(4, 4),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last',
                                   input_shape=self.ip_shape
                                   )

        self.conv2 = layers.Conv2D(filters=64,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last'
                                   )

        self.conv3 = layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last'
                                   )
        # reshape
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=512,
                                activation=tf.keras.activations.relu
                                )


        # policy output layer (Actor)
        self.policy_logits = layers.Dense(units=self.action_size, activation=tf.nn.softmax, name='policy_logits')

        # value output layer (Critic)
        self.values = layers.Dense(units=1, name='value')

    @tf.function
    def call(self, inputs):
        # converts RGB image to grayscale
        # x = tf.image.rgb_to_grayscale(inputs)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)

        logits = self.policy_logits(x)
        values = self.values(x)

        return logits, values