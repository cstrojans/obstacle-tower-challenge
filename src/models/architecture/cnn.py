"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import keras
from tensorflow.python.keras import layers


class CNN(keras.Model):
    def __init__(self, action_size, ip_shape=(84, 84, 3)):
        super(CNN, self).__init__()
        self.action_size = action_size
        self.ip_shape = ip_shape

        # common network with shared parameters
        # (20, 20, 16)
        self.conv1 = layers.Conv2D(filters=16,
                                   kernel_size=(8, 8),
                                   strides=(4, 4),
                                   padding='same',
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last',
                                   input_shape=self.ip_shape
                                   )
        # (9, 9, 32)
        self.conv2 = layers.Conv2D(filters=32,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   padding='same',
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last'
                                   )
        # (9 * 9 * 32)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(
            units=256, activation=tf.keras.activations.relu)

        # policy output layer (Actor)
        self.policy_logits = layers.Dense(
            self.action_size, name='policy_logits')

        # value output layer (Critic)
        self.values = layers.Dense(units=1, name='value')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)

        logits = self.policy_logits(x)
        values = self.values(x)

        return logits, values

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        beta_regularizer = 0.01

        if done:  # game has terminated
            reward_sum = 0.
        else:
            reward_sum = self.call(tf.convert_to_tensor(
                np.expand_dims(new_state, axis=0), dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        # TODO: try to normalize the discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.call(tf.convert_to_tensor(
            np.stack(memory.states), dtype=tf.float32))

        # get our advantages
        q_value_estimate = tf.convert_to_tensor(
            np.array(discounted_rewards)[:, None], dtype=tf.float32)
        advantage = q_value_estimate - values

        # value loss
        value_loss = advantage ** 2

        # policy loss
        actions_one_hot = tf.one_hot(
            memory.actions, self.action_size, dtype=tf.float32)

        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=policy, logits=logits)
        # entropy = tf.reduce_sum(policy * tf.math.log(policy + 1e-20), axis=1)

        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=memory.actions, logits=logits)
        # policy_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        #     labels=actions_one_hot, logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= beta_regularizer * entropy

        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))

        return total_loss
"""


import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers


class CNN(keras.Model):
    def __init__(self, action_size, ip_shape=(84, 84, 3)):
        super(CNN, self).__init__()
        self.action_size = action_size
        self.ip_shape = ip_shape

        self.conv1 = layers.Conv2D(filters=16,
                                   kernel_size=(8, 8),
                                   strides=(4, 4),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last',
                                   input_shape=self.ip_shape
                                   )
        
        self.conv2 = layers.Conv2D(filters=32,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last'
                                   )

        # reshape
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=256,
                                activation=tf.keras.activations.relu
                                )


        # policy output layer (Actor)
        self.policy_logits = layers.Dense(units=self.action_size, activation=tf.nn.softmax, name='policy_logits')

        # value output layer (Critic)
        self.values = layers.Dense(units=1, name='value')

    def call(self, inputs):
        # converts RGB image to grayscale
        # x = tf.image.rgb_to_grayscale(inputs)
        x = self.conv1(inputs)
        x = self.conv2(x)

        x = self.flatten(x)
        x = self.fc1(x)

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
            last_state = tf.convert_to_tensor(last_state)
            last_state = tf.expand_dims(last_state, axis=0)
            _, critic_value = self.call(last_state)
            discounted_reward_sum = critic_value[0, 0]
        
        returns = []
        for reward in memory.rewards_history[::-1]:  # reverse buffer r
            discounted_reward_sum = reward + gamma * discounted_reward_sum
            returns.append(discounted_reward_sum)
        returns.reverse()
        returns = np.array(returns)
        
        return returns
    
    def compute_loss(self, memory, last_state, done, gamma, eps, loss_fn):
        """ calculate actor and critic loss """
        returns = self.get_returns(memory, last_state, done, gamma, eps)
        actor_losses, critic_losses = [], []
        history = zip(memory.action_probs_history, memory.critic_value_history, returns)
        
        for action_prob, value, ret in history:
            advantage = ret - value
            actor_losses.append(tf.math.log(action_prob) * advantage)
            critic_losses.append(advantage ** 2)  # Mean Squared Error

        total_loss = (-sum(actor_losses) / len(actor_losses)) + 0.5 * sum(critic_losses)
        
        return total_loss
