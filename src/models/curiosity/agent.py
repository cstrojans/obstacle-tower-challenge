import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, layers, activations, losses
from tensorflow.keras.layers import Layer
from models.curiosity.networks import ConvGruNet, FeatureExtractor, ForwardModel, InverseModel


class TowerAgent(object):
    """ ICM model/network which is trained """
    def __init__(self, action_size, ip_shape=(84, 84, 3)):
        super(TowerAgent, self).__init__()
        self.action_size = action_size
        self.ip_shape = ip_shape

        self.actor_critic_model = ConvGruNet(self.action_size, self.ip_shape)
        self.feature_extractor = FeatureExtractor(self.ip_shape)
        self.forward_model = ForwardModel()
        self.inverse_model = InverseModel(self.action_size)
        self.distribution = tfp.distributions.Categorical
        
        self.ent_coeff = 0.001
        self.eta = 0.1
        self.value_coeff = 0.5
        self.beta = 0.8
        self.isc_lambda = 0.8
    
    def mse_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
        
    def act(self, state, training=False):
        """ get estimated policy and value """
        policy, value = self.actor_critic_model(state, training=training)

        return policy, value

    def icm_act(self, state, new_state, action_one_hot, training=False):
        """ calculate intrinsic reward 
        (state, new_state, action) -> intrinsic_reward

        Intrinsic reward calculation:
            eta/2 * mean((F'(St+1) - F(St+1))^2)
        """
        state = tf.expand_dims(state, axis=0)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state_features = self.feature_extractor(state, training=training)
        
        new_state = tf.expand_dims(new_state, axis=0)
        new_state = tf.convert_to_tensor(new_state, dtype=tf.float32)
        new_state_features = self.feature_extractor(new_state, training=training)
        
        action_one_hot = tf.cast(action_one_hot, tf.float32)
        pred_state_features = self.forward_model(state_features, action_one_hot)
        intrinsic_reward = (self.eta / 2) * self.mse_loss(pred_state_features, new_state_features)
        
        return intrinsic_reward, state_features, new_state_features

    def forward_act(self, batch_state_features, batch_action_indices):
        batch_state_features = tf.cast(batch_state_features, tf.float32)
        batch_action_indices = tf.cast(batch_action_indices, tf.float32)
        return self.forward_model(batch_state_features, batch_action_indices)

    def inverse_act(self, batch_state_features, batch_new_state_features):
        batch_state_features = tf.cast(batch_state_features, tf.float32)
        batch_new_state_features = tf.cast(batch_new_state_features, tf.float32)
        return self.inverse_model(batch_state_features, batch_new_state_features)

    def forward_loss(self, new_state_features, new_state_pred):
        """
        prediction error between predicted feature space and actual feature space of the state
        """
        forward_loss = self.mse_loss(new_state_pred, new_state_features)
        return forward_loss

    def inverse_loss(self, pred_acts, action_indices):
        """
        logits = output of inverse model - before softmax is applied
        aindex = one-hot encoding from memory
        self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=aindex), name="invloss")
        """
        action_indices = tf.concat(action_indices, axis=0)
        cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
        inverse_loss = cross_entropy_loss(action_indices, pred_acts)
        return inverse_loss

    def actor_critic_loss(self, policy, values, returns, entropy):
        actor_loss, critic_loss = 0.0, 0.0
        alpha = 0.5
        beta = 0.001
        n = len(returns)

        for policy, val, ret in zip(policy, values, returns):
            advantage = ret - val
            actor_loss = actor_loss + (-tf.math.log(tf.clip_by_value(policy, 1e-20, 1.0)) * advantage)
            critic_loss = critic_loss + (advantage ** 2)
        
        actor_loss = actor_loss / n
        critic_loss = critic_loss / n
        total_loss = actor_loss + alpha * critic_loss + beta * entropy
        return total_loss
    
    def get_returns(self, rewards):
        """
        Calculate expected value from rewards
        - At each timestep what was the total reward received after that timestep
        - Rewards in the past are discounted by multiplying them with gamma
        - These are the labels for our critic
        """
        discounted_reward_sum = 0
        gamma = 0.99
        returns = []
        for reward in rewards[::-1]:  # reverse buffer r
            discounted_reward_sum = reward + gamma * discounted_reward_sum
            returns.append(discounted_reward_sum)
        returns.reverse()

        return returns
    
    def compute_loss(self, memory, episode_reward, entropy):
        returns  = self.get_returns(memory.rewards)
        
        policy_acts, new_value = self.act(np.stack(memory.frames))
        predicted_states = self.forward_act(
            tf.concat(memory.state_features, axis=0), 
            tf.concat(memory.action_indices, axis=0))
        predicted_acts = self.inverse_act(
            tf.concat(memory.state_features, axis=0), 
            tf.concat(memory.new_state_features, axis=0))
        
        """
        policy_loss = self.policy_loss(policy_acts, advantage, memory.action_indices)
        value_loss = self.value_loss(returns, new_value)
        entropy = self.entropy(policy_acts)
        loss = policy_loss + self.value_coeff * value_loss - self.ent_coeff * entropy
        """

        ac_loss = self.actor_critic_loss(memory.policy, memory.values, returns, entropy)
        forward_loss = self.forward_loss(memory.new_state_features, predicted_states)
        inverse_loss = self.inverse_loss(predicted_acts, memory.action_indices)

        agent_loss = self.isc_lambda * ac_loss + (1 - self.beta) * inverse_loss + self.beta * forward_loss

        # print("Loss Values:\nAC Loss = {}\nEntropy = {}\nForward Loss = {}\nInverse Loss = {}\nAgent Loss = {}\n".format(
        #     ac_loss, entropy, forward_loss, inverse_loss, agent_loss))
        return ac_loss, agent_loss, forward_loss, inverse_loss
