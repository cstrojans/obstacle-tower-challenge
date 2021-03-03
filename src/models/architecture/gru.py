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
        self.policy_logits = layers.Dense(
            units=self.action_size, name='policy_logits')

        # value output layer (Critic)
        self.values = layers.Dense(units=1, name='value')

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

    def get_discounted_rewards(self, new_state, memory, done, gamma, eps):
        discounted_reward_sum = 0
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            discounted_reward_sum = reward + gamma * discounted_reward_sum
            discounted_rewards.append(discounted_reward_sum)
        discounted_rewards.reverse()

        # normalize to control the gradient estimator variance
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + eps)
        returns = tf.convert_to_tensor(discounted_rewards[:, None], dtype=tf.float32)

        # print("returns, shape = {}, type: {}".format(returns.shape, type(returns)))
        return returns

    def get_advantage(self, returns, values):
        advantage = tf.subtract(returns, values)
        # print("advantage. Shape = {}, type: {}".format(advantage.shape, type(advantage)))
        # advantage = (advantage - tf.math.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-6) # advantage normalization
        return advantage

    def policy_loss(self, memory, policy, advantage):
        """
        A2C policy loss calculation: -1/n * sum(advantage * log(policy)).
        """
        # policy_logs = tf.math.log(tf.clip_by_value(policy, clip_value_min=1e-20, clip_value_max=1.0))
        policy_logs = tf.math.log(policy)

        # only take policies for taken actions
        action_indices = tf.one_hot(memory.actions, self.action_size, dtype=tf.float32)
        pi_logs = tf.math.reduce_sum(tf.math.multiply(policy_logs, action_indices), axis=1)
        policy_loss = -tf.math.reduce_mean(advantage * pi_logs) / len(memory.actions)
        
        return policy_loss
    
    def value_loss(self, returns, values):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(values, returns).numpy()
    
    def actor_loss(self, advantage, policy):
        actor_loss = -tf.math.reduce_mean(advantage * policy)
        print("policy_loss: {}, Shape = {}, type: {}".format(actor_loss, actor_loss.shape, type(actor_loss)))
        return actor_loss
    
    def critic_loss(self, returns, values):
        # mse = tf.keras.losses.MeanSquaredError()
        # critic_loss = mse(memory.critic_value_history, discounted_rewards).numpy()

        huber_loss = tf.keras.losses.Huber()
        critic_loss = huber_loss(values, returns).numpy()
        print("value_loss: {}, type: {}".format(critic_loss, type(critic_loss)))
        return critic_loss


    def entropy_loss(self, policy):
        dist = tfp.distributions.Categorical
        return dist(probs=policy).entropy()

    def compute_loss(self, done, new_state, memory, gamma, eps):
        # compiles, but loss keeps increasing
        returns = self.get_discounted_rewards(new_state, memory, done, gamma, eps) # (65, 1)
        
        policy_logits, values = self.call(tf.convert_to_tensor(np.stack(memory.states), dtype=tf.float32))
        policy = tf.nn.softmax(policy_logits)
        # print("policy type: Shape = {}, {}".format(policy.shape, type(policy)))
        # print("values type: Shape = {}, {}".format(values.shape, type(values)))
        
        advantage = self.get_advantage(returns, values)
        # print("advantage Shape = {}, type: {}".format(advantage.shape, type(advantage)))
        
        value_loss = self.value_loss(returns, values)
        # print("value_loss: {}, type: {}".format(value_loss, type(value_loss)))
        
        policy_loss = self.policy_loss(memory, policy, advantage)
        # print("policy_loss: {}, Shape = {}, type: {}".format(policy_loss, policy_loss.shape, type(policy_loss)))
        
        entropy_loss = self.entropy_loss(policy)
        # print("entropy_loss: {}, Shape = {}, type: {}".format(entropy_loss, entropy_loss.shape, type(entropy_loss)))

        value_coeff = 0.5
        entropy_coeff = 0.001
        total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy_loss
        # print("total_loss: {}, Shape = {}, type: {}".format(total_loss, total_loss.shape, type(total_loss)))

        return total_loss
        
        """
        # not working setup
        returns = self.get_discounted_rewards(new_state, memory, done, gamma, eps)
        policy_logits = np.array(memory.action_probs_history)
        values = np.array(memory.critic_value_history)
        print("policy type: Shape = {}, {}".format(policy_logits.shape, type(policy_logits)))
        print("values type: Shape = {}, {}".format(values.shape, type(values)))

        advantage = self.get_advantage(returns, memory.critic_value_history)
        
        actor_loss = self.actor_loss(advantage, memory.action_probs_history)
        critic_loss = self.critic_loss(returns, memory.critic_value_history)
        total_loss = actor_loss + critic_loss
        
        print("Total Loss: {}, type = {}".format(total_loss, type(total_loss)))
        return total_loss
        """