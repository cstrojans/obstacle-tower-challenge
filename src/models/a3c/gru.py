import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import keras
from tensorflow.keras import layers


class CnnGru(keras.Model):
    def __init__(self, action_size, ip_shape=(84, 84, 3)):
        super(CnnGru, self).__init__()
        self.action_size = action_size
        self.ip_shape = ip_shape

        # CNN - spatial dependencies
        # (20, 20, 32)
        self.conv1 = layers.Conv2D(filters=32,
                                   kernel_size=(8, 8),
                                   strides=(4, 4),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last',
                                   input_shape=self.ip_shape
                                   )
        self.bn1 = layers.BatchNormalization()
        # self.pool1 = layers.MaxPool2D(pool_size=(2, 2))

        # (9, 9, 64)
        self.conv2 = layers.Conv2D(filters=64,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last'
                                   )
        self.bn2 = layers.BatchNormalization()
        # self.pool2 = layers.MaxPool2D(pool_size=(2, 2))

        # (7, 7, 64)
        self.conv3 = layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last'
                                   )
        self.bn3 = layers.BatchNormalization()

        # reshape
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=512,
                                activation=tf.keras.activations.relu
                                )

        # RNN - temporal dependencies
        self.gru = layers.GRU(512, activation=tf.keras.activations.tanh)

        # policy output layer (Actor)
        self.policy_logits = layers.Dense(units=self.action_size, activation=tf.nn.softmax, name='policy_logits')

        # value output layer (Critic)
        self.values = layers.Dense(units=1, name='value')

    @tf.function
    def call(self, inputs, training=False):
        state, rem_time = inputs[0], inputs[1]
        # state = state / 255.0
        x = tf.image.rgb_to_grayscale(state)
        rem_time = tf.expand_dims(tf.expand_dims(rem_time, axis=0), axis=1)  # (1, 1)
        
        x = self.conv1(state)
        x = self.bn1(x, training=training)
        # x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        # x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.flatten(x)  # (1, 3136)
        x = tf.concat([x, rem_time], axis=1)  # (1, 3137)
        x = self.fc1(x)

        # input: [batch, timesteps, feature]
        x = tf.expand_dims(x, axis=0)
        x = self.gru(x)

        logits = self.policy_logits(x)
        values = self.values(x)

        return logits, values

    def get_returns(self, memory, last_state, done, gamma, eps):
        """
        Calculate expected value from rewards
        - At each timestep what was the total reward received after that timestep
        - Rewards in the past are discounted by multiplying them with gamma
        - These are the labels for our critic
        """
        if done:  # game has terminated
            discounted_reward_sum = 0.
        else:  # bootstrap starting reward from last state
            discounted_reward_sum = memory.critic_value_history[-1]

        returns = []
        for reward in memory.rewards_history[::-1]:  # reverse buffer r
            discounted_reward_sum = reward + gamma * discounted_reward_sum
            returns.append(discounted_reward_sum)
        returns.reverse()
        return returns
    
    def compute_loss(self, memory, last_state, done, gamma, eps, entropy):
        """ calculate actor and critic loss """
        value_coeff = 0.5
        entropy_coeff = 0.01
        returns = self.get_returns(memory, last_state, done, gamma, eps)

        # advantage: 
        # how much better it is to take a specific action compared to 
        # the average, general action at the given state.
        advantage = tf.math.subtract(returns, memory.critic_value_history)
        actor_loss = tf.math.reduce_mean(tf.stop_gradient(advantage) * memory.action_probs_history)
        critic_loss = tf.keras.losses.MSE(returns, memory.critic_value_history).numpy()

        total_loss = actor_loss + value_coeff * critic_loss + entropy_coeff * entropy        
        return -total_loss  # negate it to perform gradient ascent
