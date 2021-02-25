import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import keras
from tensorflow.python.keras import layers
from models.util import normalized_columns_initializer


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
                                   padding='same',
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last',
                                   input_shape=self.ip_shape
                                   )

        # (9, 9, 64)
        self.conv2 = layers.Conv2D(filters=64,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   padding='same',
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last'
                                   )

        # (7, 7, 64)
        self.conv3 = layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding='same',
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last'
                                   )

        # reshape
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=512,
                                activation=tf.keras.activations.relu
                                )

        # RNN - temporal dependencies
        self.gru = layers.GRU(512)

        # policy output layer (Actor)
        self.policy_logits = layers.Dense(units=self.action_size,
                                          kernel_initializer=normalized_columns_initializer(
                                              0.01),
                                          name='policy_logits'
                                          )

        # value output layer (Critic)
        self.values = layers.Dense(units=1,
                                   kernel_initializer=normalized_columns_initializer(
                                       1.0),
                                   name='value')

    def call(self, inputs):
        # converts RGB image to grayscale
        x = tf.image.rgb_to_grayscale(inputs)
        # x = tf.image.per_image_standardization(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)
        x = self.fc1(x)

        # input: [batch, timesteps, feature]
        x = tf.expand_dims(x, axis=0)
        x = self.gru(x)

        logits = self.policy_logits(x)
        values = self.values(x)

        return logits, values

    def get_discounted_rewards(self, new_state, memory, done, gamma):
        if done:  # game has terminated
            reward_sum = 0.
        else:
            reward_sum = self.call(tf.convert_to_tensor(
                np.expand_dims(new_state, axis=0), dtype=tf.float32))[-1].numpy()[0]

        # TODO: try to normalize the discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        return discounted_rewards

    def get_advantage(self, returns, values):
        advantage = returns - values
        advantage = (advantage - tf.math.reduce_mean(advantage)) / \
            (tf.math.reduce_std(advantage) + 1e-6)
        return advantage

    def policy_loss(self, memory, policy, advantage):
        """
        A2C policy loss calculation: -1/n * sum(advantage * log(policy)).
        """
        policy_logs = tf.math.log(tf.clip_by_value(
            policy, clip_value_min=1e-20, clip_value_max=1.0))

        # only take policies for taken actions
        action_indices = tf.one_hot(
            memory.actions, self.action_size, dtype=tf.float32)
        pi_logs = tf.math.reduce_sum(tf.math.multiply(
            policy_logs, action_indices), axis=1)
        policy_loss = -tf.math.reduce_mean(advantage * pi_logs)

        return policy_loss

    def value_loss(self, returns, values):
        mse = tf.keras.losses.MeanSquaredError()
        return 0.5 * mse(values, returns).numpy()

    def entropy(self, policy):
        dist = tfp.distributions.Categorical
        return dist(probs=policy).entropy()

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        discounted_rewards = self.get_discounted_rewards(
            new_state, memory, done, gamma)
        returns = tf.convert_to_tensor(np.array(discounted_rewards)[
                                       :, None], dtype=tf.float32)

        policy_logits, values = self.call(
            tf.convert_to_tensor(np.stack(memory.states), dtype=tf.float32))
        policy = tf.nn.softmax(policy_logits)
        advantage = self.get_advantage(returns, values)

        value_loss = self.value_loss(returns, values)
        policy_loss = self.policy_loss(memory, policy, advantage)
        entropy = self.entropy(policy)

        value_coeff = 0.5
        entropy_coeff = 0.01
        total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

        return total_loss
