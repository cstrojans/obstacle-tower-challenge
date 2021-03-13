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
        self.conv1 = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(
            4, 4), padding='same', activation='relu', data_format='channels_last', input_shape=self.ip_shape)
        # (9, 9, 32)
        self.conv2 = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(
            2, 2), padding='same', activation='relu', data_format='channels_last')
        # (9 * 9 * 32)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(units=256, activation='relu')

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

    def get_layer_weights(self):
        weights = []
        weights.append(self.conv1.get_weights())
        weights.append(self.conv2.get_weights())
        weights.append(self.flatten.get_weights())
        weights.append(self.dense1.get_weights())
        weights.append(self.policy_logits.get_weights())
        weights.append(self.values.get_weights())

        return weights

    def put_layer_weights(self, weights):
        self.conv1.set_weights(weights[0])
        self.conv2.set_weights(weights[1])
        self.flatten.set_weights(weights[2])
        self.dense1.set_weights(weights[3])
        self.policy_logits.set_weights(weights[4])
        self.values.set_weights(weights[5])

    
    def compute_loss_ppo(self, done, new_state, memory, gamma, prev_logprobs, prev_CNN):
        """
        Write down the steps
        1. Need to pass the following things: 
            done: If game has terminated
            new_state: for calculating next reward
            memory
            gamma
            clipping_val
            entropy_beta
            lambda
        2. Calculate next value and add to memory.values
        3. Calculate advantages
            - 
        4. Calculate ratio r (newprobabilities / old)
        5. 
        """
        # beta_regularizer = 0.01
        clipping_val = 0.2
        critic_discount = 0.5
        entropy_beta = 0.001
        gamma = 0.99
        lmbda = 0.95

        
        if done:  # game has terminated
            reward_sum = 0
        else:
            last_state = tf.convert_to_tensor(tf.expand_dims(new_state, axis=0), dtype=tf.float32)
            _, critic_val = self.call(last_state)
            print(critic_val)
            reward_sum = critic_val[0, 0]

        memory.values.append(reward_sum) 
        values = memory.values
        masks = memory.is_terminals
        rewards = memory.rewards       
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + gamma * lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])
        
        new_logits, new_values = self.call(tf.convert_to_tensor(
            np.stack(memory.states), dtype=tf.float32))
        
        old_logits, old_values = prev_CNN.call(tf.convert_to_tensor(
            np.stack(memory.states), dtype=tf.float32))
        
        q_value_estimate = tf.convert_to_tensor(
            np.array(returns)[:, None], dtype=tf.float32)
        
        advantage = q_value_estimate - new_values

        # value loss
        value_loss = advantage ** 2
        # _, advantage = self.get_advantages(memory, gamma, lmbda)
        # advantage = tf.convert_to_tensor(advantage)

        # Try to understand flow here
        # My question is: We will get initial probabilities by interacting with the environment
        # But we won't get next probabilities till we update the model
        # So for iteration 1, how do we find r?
        # Possible soln: Think in terms of y_true and y_predict
        # y_predict: actions that we get from policy output
        # y_true: actions that were actually taken!!
        # So basically, you will need to keep track of old weights
        
        # new_policy_probs = tf.nn.softmax(logits)
        # new_policy_probs = np.array(new_policy_probs)
        
        new_policy = tf.nn.softmax(new_logits)
        old_policy = tf.nn.softmax(old_logits)
        
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=new_policy, logits=new_logits)
        
        ratio = tf.math.exp(tf.math.log(new_policy + 1e-10) - tf.math.log(old_policy + 1e-10))

        p1 = ratio * tf.stop_gradient(advantage)
        p2 = tf.clip_by_value(ratio, 1 - clipping_val, 1 + clipping_val) * tf.stop_gradient(advantage)
        policy_loss = -tf.math.reduce_mean(tf.math.minimum(p1, p2))
        total_loss = critic_discount * value_loss + policy_loss - entropy_beta * entropy
        

        return tf.reduce_mean(total_loss)

    def get_advantages(self, memory, gamma, lmbda):
        values = memory.values
        masks = memory.is_terminals
        rewards = memory.rewards
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + gamma * lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    
    # def get_returns(self, memory, last_state, done, gamma, eps):
    #     """
    #     Calculate expected value from rewards
    #     - At each timestep what was the total reward received after that timestep
    #     - Rewards in the past are discounted by multiplying them with gamma
    #     - These are the labels for our critic
    #     """
    #     if done:  # game has terminated
    #         discounted_reward_sum = 0.
    #     else:  # bootstrap starting reward from last state
    #         last_state = tf.convert_to_tensor(last_state)
    #         last_state = tf.expand_dims(last_state, axis=0)
    #         _, critic_value = self.call(last_state)
    #         discounted_reward_sum = critic_value[0, 0]
        
    #     returns = []
    #     for reward in memory.rewards_history[::-1]:  # reverse buffer r
    #         discounted_reward_sum = reward + gamma * discounted_reward_sum
    #         returns.append(discounted_reward_sum)
    #     returns.reverse()
    #     returns = np.array(returns)
        
    #     return returns
