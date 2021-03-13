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
        self.conv1 = layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding='same', activation=lambda x: tf.nn.leaky_relu(
            x, alpha=0.01), data_format='channels_last', input_shape=self.ip_shape)

        # (9, 9, 64)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(
            2, 2), padding='same', activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01), data_format='channels_last')

        # (7, 7, 64)
        self.conv3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(
            1, 1), padding='same', activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01), data_format='channels_last')

        # reshape
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(
            units=512, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01))

        # RNN - temporal dependencies
        self.gru = layers.GRU(512)

        # policy output layer (Actor)
        self.policy_logits = layers.Dense(
            self.action_size, name='policy_logits')

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
    
    def get_layer_weights(self):
        weights = []
        weights.append(self.conv1.get_weights())
        weights.append(self.conv2.get_weights())
        weights.append(self.conv3.get_weights())
        weights.append(self.flatten.get_weights())
        weights.append(self.fc1.get_weights())
        weights.append(self.gru.get_weights())
        weights.append(self.policy_logits.get_weights())
        weights.append(self.values.get_weights())

        return weights

    def put_layer_weights(self, weights):
        self.conv1.set_weights(weights[0])
        self.conv2.set_weights(weights[1])
        self.conv3.set_weights(weights[2])
        self.flatten.set_weights(weights[3])
        self.fc1.set_weights(weights[4])
        self.gru.set_weights(weights[5])
        self.policy_logits.set_weights(weights[6])
        self.values.set_weights(weights[7])
    
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
        # return 0.5 * mse(values, returns).numpy()
        return mse(values, returns).numpy()

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

    def compute_loss_ppo(self, done, new_state, memory, gamma, prev_logprobs, prev_CNN):
        clipping_val = 0.2
        critic_discount = 0.5
        entropy_beta = 0.001
        gamma = 0.99
        lmbda = 0.95


        if done:  # game has terminated
            reward_sum = 0.
        else:
            reward_sum = self.call(tf.convert_to_tensor(
                np.expand_dims(new_state, axis=0), dtype=tf.float32))[-1].numpy()[0]

        memory.values.append(reward_sum) 
        values = memory.values
        masks = memory.is_terminals
        rewards = memory.rewards       
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
            # print(rewards[i], values[i].shape, delta.shape)
            gae = delta + gamma * lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])
        
        new_logits, new_values = self.call(tf.convert_to_tensor(
            np.stack(memory.states), dtype=tf.float32))
        
        old_logits, old_values = prev_CNN.call(tf.convert_to_tensor(
            np.stack(memory.states), dtype=tf.float32))
        
        q_value_estimate = tf.convert_to_tensor(
            np.array(returns)[:, None], dtype=tf.float32)
        
        advantage = self.get_advantage(q_value_estimate, new_values)

        value_loss = self.value_loss(q_value_estimate, new_values)
        # value_loss = 

        new_policy = tf.nn.softmax(new_logits)
        old_policy = tf.nn.softmax(old_logits)
        
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=new_policy, logits=new_logits)

        ratio = tf.math.exp(tf.math.log(new_policy + 1e-10) - tf.math.log(old_policy + 1e-10))

        p1 = ratio * tf.stop_gradient(advantage)
        p2 = tf.clip_by_value(ratio, 1 - clipping_val, 1 + clipping_val) * tf.stop_gradient(advantage)
        policy_loss = -tf.math.reduce_mean(tf.math.minimum(p1, p2))
        total_loss = critic_discount * value_loss + policy_loss - entropy_beta * entropy
        
        
        return tf.reduce_mean(total_loss)
