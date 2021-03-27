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
        self.conv1 = layers.Conv2D(filters=16,
                                   kernel_size=(8, 8),
                                   strides=(4, 4),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last',
                                   input_shape=self.ip_shape
                                   )

        # (9, 9, 64)
        self.conv2 = layers.Conv2D(filters=32,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last'
                                   )

        # (7, 7, 64)
        self.conv3 = layers.Conv2D(filters=16,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   activation=tf.keras.activations.relu,
                                   data_format='channels_last'
                                   )

        # reshape
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=128,
                                activation=tf.keras.activations.relu
                                )

        # RNN - temporal dependencies
        self.gru = layers.GRU(128)

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
            # critic_losses.append(loss_fn(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))  # Huber Loss

        total_loss = (-sum(actor_losses) / len(actor_losses)) + 0.5 * sum(critic_losses)
        
        return total_loss
