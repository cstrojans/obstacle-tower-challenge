import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, activations, losses
from tensorflow.keras.layers import Layer


class ConvGruNet(Layer):
    """ Actor Critic Model """
    def __init__(self, action_size, ip_shape=(84, 84, 3), **kwargs):
        super(ConvGruNet, self).__init__(**kwargs)
        self.action_size = action_size
        self.ip_shape = ip_shape

        self.conv1 = layers.Conv2D(filters=32,
                                   kernel_size=(8, 8),
                                   strides=(4, 4),
                                   activation=layers.LeakyReLU(alpha=0.01),
                                   data_format='channels_last',
                                   input_shape=self.ip_shape
                                   )
        self.bn1 = layers.BatchNormalization()

        # (9, 9, 64)
        self.conv2 = layers.Conv2D(filters=64,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   activation=layers.LeakyReLU(alpha=0.01),
                                   data_format='channels_last'
                                   )
        self.bn2 = layers.BatchNormalization()

        # (7, 7, 64)
        self.conv3 = layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   activation=layers.LeakyReLU(alpha=0.01),
                                   data_format='channels_last'
                                   )
        self.bn3 = layers.BatchNormalization()

        # reshape
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=512,
                                activation=layers.LeakyReLU(alpha=0.01)
                                )
        
        # LSTM component
        # models the sequence on agents' movements
        # an alternate to frame stacking
        self.gru = layers.GRU(512, return_state=True)

        # Actor - policy output layer
        # chooses the best action to perform in each timestep
        self.policy_logits = layers.Dense(units=action_size, activation=tf.nn.softmax, name='policy_logits')

        # Critic - value output layer
        # gives the value of an action as feedback to the Actor
        self.values = layers.Dense(units=1, name='value')
    
    def call(self, state, training=False):
        # feature maps using convolutional layers
        features = self.conv1(state)
        features = self.bn1(features, training=training)
        features = self.conv2(features)
        features = self.bn2(features, training=training)
        features = self.conv3(features)
        features = self.bn3(features, training=training)
        features = self.flatten(features)
        features = self.fc1(features)

        # input: [batch, timesteps, feature]
        features = tf.expand_dims(features, axis=0)
        features, hidden_state = self.gru(features)

        # Actor-Critic Outputs
        policy = self.policy_logits(features)
        value = self.values(features)

        return policy, value

class FeatureExtractor(Layer):
    """ learns a feature representation of the state """
    def __init__(self, ip_shape, **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs)
        self.ip_shape = ip_shape

        self.conv1 = layers.Conv2D(filters=32,
                                   kernel_size=(8, 8),
                                   strides=(4, 4),
                                   activation=layers.LeakyReLU(alpha=0.01),
                                   data_format='channels_last',
                                   input_shape=self.ip_shape
                                   )
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filters=64,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   activation=layers.LeakyReLU(alpha=0.01),
                                   data_format='channels_last'
                                   )
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   activation=layers.LeakyReLU(alpha=0.01),
                                   data_format='channels_last'
                                   )
        self.bn3 = layers.BatchNormalization()

        # reshape
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=288, activation=layers.LeakyReLU(alpha=0.01))
    
    def call(self, state, training=False):
        # TODO: observation normalization before passing to conv1 layer
        pred_state_f = self.conv1(state)
        pred_state_f = self.bn1(pred_state_f, training=training)
        pred_state_f = self.conv2(pred_state_f)
        pred_state_f = self.bn2(pred_state_f, training=training)
        pred_state_f = self.conv3(pred_state_f)
        pred_state_f = self.bn3(pred_state_f, training=training)
        pred_state_f = self.flatten(pred_state_f)
        pred_state_f = self.fc1(pred_state_f)
        return pred_state_f

class ForwardModel(Layer):
    def __init__(self, **kwargs):
        super(ForwardModel, self).__init__(**kwargs)
        self.fc1 = layers.Dense(units=256, activation=layers.LeakyReLU(alpha=0.01))
        self.hidden_1 = layers.Dense(units=256*2, activation=layers.LeakyReLU(alpha=0.01))
        self.hidden_2 = layers.Dense(units=288, activation=layers.LeakyReLU(alpha=0.01))
    
    def call(self, action_one_hot, state_features):
        concat_features = tf.concat([state_features, action_one_hot], axis=1)
        pred_next_state_f = self.fc1(concat_features)
        pred_next_state_f = self.hidden_1(pred_next_state_f)
        pred_next_state_f = self.hidden_2(pred_next_state_f)
        return pred_next_state_f

class InverseModel(Layer):
    def __init__(self, action_size, **kwargs):
        super(InverseModel, self).__init__(**kwargs)
        self.fc1 = layers.Dense(units=256, activation=layers.LeakyReLU(alpha=0.01))
        self.hidden_1 = layers.Dense(units=256*2, activation=layers.LeakyReLU(alpha=0.01))
        self.hidden_2 = layers.Dense(units=256*2, activation=layers.LeakyReLU(alpha=0.01))
        self.op = layers.Dense(units=action_size, activation=tf.nn.softmax)
    
    def call(self, state_features, next_state_features):
        concat_features = tf.concat([state_features, next_state_features], axis=1)
        pred_action_index = self.fc1(concat_features)
        pred_action_index = self.hidden_1(pred_action_index)
        pred_action_index = self.hidden_2(pred_action_index)
        pred_action_index = self.op(pred_action_index)
        return pred_action_index
