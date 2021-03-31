import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import keras
from tensorflow.python.keras import layers


class CnnGru(keras.Model):
    def __init__(self, action_size, ip_shape=(84, 84, 3)):
        super(CnnGru, self).__init__()
        self.action_size = action_size
        self.ip_shape = ip_shape

        # CNN - spatial dependencies
        # (20, 20, 32)
        self.conv1 = layers.Conv2D(filters=16,
                                   kernel_size=(8, 8),
                                   strides=(4, 4),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last',
                                   input_shape=self.ip_shape
                                   )
        self.bn1 = layers.BatchNormalization()

        # (9, 9, 64)
        self.conv2 = layers.Conv2D(filters=32,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last'
                                   )
        self.bn2 = layers.BatchNormalization()

        # (7, 7, 64)
        self.conv3 = layers.Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last'
                                   )
        self.bn3 = layers.BatchNormalization()

        # reshape
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=256,
                                activation=tf.keras.activations.relu
                                )

        # RNN - temporal dependencies
        self.gru = layers.LSTM(256)

        # policy output layer (Actor)
        self.policy_logits = layers.Dense(units=self.action_size, activation=tf.nn.softmax, name='policy_logits')

        # value output layer (Critic)
        self.values = layers.Dense(units=1, name='value')

    @tf.function
    def call(self, inputs, training=False):
        # converts RGB image to grayscale
        # x = tf.image.rgb_to_grayscale(inputs)
        x = inputs / 255.0
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.flatten(x)
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

        """
        if done:  # game has terminated
            discounted_reward_sum = 0.
        else:  # bootstrap starting reward from last state
            last_state = tf.convert_to_tensor(last_state)
            last_state = tf.expand_dims(last_state, axis=0)
            _, critic_value = self.call(last_state)
            discounted_reward_sum = critic_value[0, 0]
        """

        discounted_reward_sum = 0
        returns = []
        for reward in memory.rewards_history[::-1]:  # reverse buffer r
            discounted_reward_sum = reward + gamma * discounted_reward_sum
            returns.append(discounted_reward_sum)
        returns.reverse()
        return returns
    
    def compute_loss(self, memory, last_state, done, gamma, eps, entropy):
        """ calculate actor and critic loss """
        alpha = 0.5
        beta = 0.001
        returns = self.get_returns(memory, last_state, done, gamma, eps)
        actor_loss, critic_loss = 0.0, 0.0
        history = zip(memory.action_probs_history, memory.critic_value_history, returns)
        n = len(returns)

        for action_prob, value, ret in history:
            # advantage: how much better it is to take a specific action compared to 
            # the average, general action at the given state.
            advantage = ret - value
            actor_loss = actor_loss + (-tf.math.log(action_prob) * advantage)
            critic_loss = critic_loss + (advantage ** 2)

        actor_loss = actor_loss / n
        critic_loss = critic_loss / n
        total_loss = actor_loss + alpha * critic_loss + beta * entropy
        
        return total_loss
