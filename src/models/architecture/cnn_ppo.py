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

    def get_returns(self, memory, last_state, done, gamma, eps, lmbda):
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
        # discounted_reward_sum = 1
        memory.values.append(discounted_reward_sum) 
        values = memory.values
        masks = memory.is_terminals
        rewards = memory.rewards       
        returns = []
        a_t = 0
        for i in range(len(rewards)):
            discount = 1
            a_t = 0
            for k in range(i, len(rewards)):
                a_t += discount * (rewards[i] + gamma * values[i + 1] \
                    * (1 - masks[i]) - values[i])
                # a_t += (rewards[i] + gamma * values[i + 1] * (1 - masks[i]) - values[i])
                discount *= gamma * lmbda
            returns.append(a_t)
        
        return returns
    
    def compute_loss(self, memory, last_state, done, gamma, eps, loss_fn):
        """ calculate actor and critic loss """
        clipping_val = 0.2
        critic_discount = 0.5
        entropy_beta = 0.001
        gamma = 0.99
        lmbda = 0.95
        
        returns = self.get_returns(memory, last_state, done, gamma, eps, lmbda)
        actor_losses, critic_losses, entropy = [], [], []
        history = zip(memory.action_probs, memory.prev_action_probs, memory.values, returns)
        
        for action_prob, prev_action_prob, value, ret in history:
            advantage = ret - value
            r = math.exp(math.log(action_prob) - math.log(prev_action_prob))
            p1 = r * advantage
            p2 = max(min(r, 1 + clipping_val), 1 - clipping_val) * advantage

            actor_losses.append(min(p1, p2))
            critic_losses.append(advantage ** 2)  # Mean Squared Error
            entropy.append(-(action_prob * math.log(action_prob)))
            # critic_losses.append(loss_fn(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))  # Huber Loss

        total_loss = (-sum(actor_losses) / len(actor_losses)) + critic_discount * sum(critic_losses) / len(critic_losses) \
            - entropy_beta * sum(entropy) / len(entropy)
        
        return total_loss