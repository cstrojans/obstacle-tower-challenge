import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Layer


class ConvGruNet(Model):
    """ Actor Critic Model """

    def __init__(self, action_size, ip_shape=(84, 84, 3), **kwargs):
        super(ConvGruNet, self).__init__(**kwargs)
        self.action_size = action_size
        self.ip_shape = ip_shape
        self.regularizer = tf.keras.regularizers.l2(1e-10)

        self.conv1 = layers.Conv2D(filters=32,
                                   kernel_size=(8, 8),
                                   strides=(4, 4),
                                   activation=tf.keras.activations.relu,
                                   kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer,
                                   data_format='channels_last',
                                   input_shape=self.ip_shape
                                   )
        # self.bn1 = layers.BatchNormalization()

        # (9, 9, 64)
        self.conv2 = layers.Conv2D(filters=64,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   activation=tf.keras.activations.relu,
                                   kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer,
                                   data_format='channels_last'
                                   )
        # self.bn2 = layers.BatchNormalization()

        # (7, 7, 64)
        self.conv3 = layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   activation=tf.keras.activations.relu,
                                   kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer,
                                   data_format='channels_last'
                                   )
        # self.bn3 = layers.BatchNormalization()

        # reshape
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=512, 
                                activation=tf.keras.activations.relu,
                                kernel_regularizer=self.regularizer,
                                bias_regularizer=self.regularizer)
        # self.fc2 = layers.Dense(units=256, activation=tf.keras.activations.relu)

        # models the sequence on agents' movements an alternate to frame stacking
        self.gru = layers.GRU(512)

        # Actor - policy output layer
        # chooses the best action to perform in each timestep
        # self.policy_dense = layers.Dense(units=128, 
        #                         activation=tf.keras.activations.relu,
        #                         kernel_regularizer=self.regularizer,
        #                         bias_regularizer=self.regularizer)
        self.policy_logits = layers.Dense(units=action_size, 
                                activation=tf.nn.softmax, 
                                name='policy_logits')

        # Critic - value output layer
        # gives the value of an action as feedback to the Actor
        # self.value_dense = layers.Dense(units=128, 
        #                         activation=tf.keras.activations.relu,
        #                         kernel_regularizer=self.regularizer,
        #                         bias_regularizer=self.regularizer)
        self.values = layers.Dense(units=1, 
                                name='value')

    @tf.function
    def call(self, state, training=False):
        state = state / 255.0
        # state = tf.image.rgb_to_grayscale(state)

        # feature maps using convolutional layers
        features = self.conv1(state)
        # features = self.bn1(features, training=training)
        features = self.conv2(features)
        # features = self.bn2(features, training=training)
        features = self.conv3(features)
        # features = self.bn3(features, training=training)

        features = self.flatten(features)
        features = self.fc1(features)

        # recurrent input: [batch, timesteps, feature]
        features = tf.expand_dims(features, axis=0)
        features = self.gru(features)

        # Actor
        # policy = self.policy_dense(features)
        policy = self.policy_logits(features)

        # Critic
        # value = self.value_dense(features)
        value = self.values(features)

        return policy, value


class FeatureExtractor(Model):
    """ learns a feature representation of the state """

    def __init__(self, ip_shape, **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs)
        self.ip_shape = ip_shape
        self.regularizer = tf.keras.regularizers.l2(1e-10)

        self.conv1 = layers.Conv2D(filters=32,
                                   kernel_size=(8, 8),
                                   strides=(4, 4),
                                   activation=tf.keras.activations.relu,
                                   kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer,
                                   data_format='channels_last',
                                   input_shape=self.ip_shape
                                   )
        # self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filters=64,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   activation=tf.keras.activations.relu,
                                   kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer,
                                   data_format='channels_last'
                                   )
        # self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   activation=tf.keras.activations.relu,
                                   kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer,
                                   data_format='channels_last'
                                   )
        # self.bn3 = layers.BatchNormalization()

        # reshape
        self.flatten = layers.Flatten()
        # self.fc1 = layers.Dense(units=288,
        #                         # activation=layers.LeakyReLU(alpha=0.01)
        #                         activation=tf.keras.activations.relu
        #                         )

    @tf.function
    def call(self, state, training=False):
        state = state / 255.0
        # state = tf.image.rgb_to_grayscale(state)

        pred_state_f = self.conv1(state)
        # pred_state_f = self.bn1(pred_state_f, training=training)
        pred_state_f = self.conv2(pred_state_f)
        # pred_state_f = self.bn2(pred_state_f, training=training)
        pred_state_f = self.conv3(pred_state_f)
        # pred_state_f = self.bn3(pred_state_f, training=training)
        pred_state_f = self.flatten(pred_state_f)
        # pred_state_f = self.fc1(pred_state_f)

        return pred_state_f


class ForwardModel(Model):
    def __init__(self, **kwargs):
        super(ForwardModel, self).__init__(**kwargs)
        self.regularizer = tf.keras.regularizers.l2(1e-10)

        self.fc1 = layers.Dense(units=512,
                                activation=tf.keras.activations.relu,
                                kernel_regularizer=self.regularizer,
                                bias_regularizer=self.regularizer
                                )
        self.hidden_1 = layers.Dense(units=512,
                                     activation=tf.keras.activations.relu,
                                     kernel_regularizer=self.regularizer,
                                     bias_regularizer=self.regularizer
                                     )
        self.hidden_2 = layers.Dense(units=3136,
                                     activation=tf.keras.activations.relu,
                                     kernel_regularizer=self.regularizer,
                                     bias_regularizer=self.regularizer
                                     )

    @tf.function
    def call(self, inputs):
        action_one_hot, state_features = inputs
        concat_features = tf.concat([state_features, action_one_hot], axis=1)
        pred_next_state_f = self.fc1(concat_features)
        pred_next_state_f = self.hidden_1(pred_next_state_f)
        pred_next_state_f = self.hidden_2(pred_next_state_f)
        return pred_next_state_f


class InverseModel(Model):
    def __init__(self, action_size, **kwargs):
        super(InverseModel, self).__init__(**kwargs)
        self.regularizer = tf.keras.regularizers.l2(1e-10)

        self.fc1 = layers.Dense(units=256,
                                activation=tf.keras.activations.relu,
                                kernel_regularizer=self.regularizer,
                                bias_regularizer=self.regularizer
                                )
        # self.hidden_1 = layers.Dense(units=256*2,
        #                              # activation=layers.LeakyReLU(alpha=0.01)
        #                              activation=tf.keras.activations.relu
        #                              )
        # self.hidden_2 = layers.Dense(units=256*2,
        #                              # activation=layers.LeakyReLU(alpha=0.01)
        #                              activation=tf.keras.activations.relu
        #                              )
        self.op = layers.Dense(units=action_size)

    @tf.function
    def call(self, inputs):
        state_features, next_state_features = inputs
        concat_features = tf.concat(
            [state_features, next_state_features], axis=1)

        pred_action_index = self.fc1(concat_features)
        # pred_action_index = self.hidden_1(pred_action_index)
        # pred_action_index = self.hidden_2(pred_action_index)
        pred_action_index = self.op(pred_action_index)

        return pred_action_index
