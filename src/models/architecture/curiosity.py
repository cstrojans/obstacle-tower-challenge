import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, layers, activations, losses
from tensorflow.keras.layers import Layer
from models.architecture.curiosity_networks import ConvGruNet, FeatureExtractor, ForwardModel, InverseModel


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
        self.value_coeff = 0.5
        self.beta = 0.8
        self.isc_lambda = 0.8
    
    def mse_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
        """
        loss = 0.0
        # y_true = np.squeeze(y_true)
        # y_pred = np.squeeze(y_pred)

        for t, p in zip(y_true, y_pred):
            loss += (t-p) ** 2
        loss /= len(y_true)
        """
        return loss
    
    def act(self, state, training=False):
        """ get estimated policy and value """
        policy, value = self.actor_critic_model(state, training=training)

        return policy, value

    def icm_act(self, state, new_state, action_one_hot, eta=0.1, training=False):
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
        
        pred_state_features = self.forward_model(state_features, action_one_hot)
        intrinsic_reward = (eta / 2) * self.mse_loss(pred_state_features, new_state_features)
        
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
        # forward_loss = 0.5 * self.mse_loss(new_state_pred, new_state_features)
        forward_loss = self.mse_loss(new_state_pred, new_state_features)
        return forward_loss

    def inverse_loss(self, pred_acts, action_indices):
        """
        logits = output of inverse model - before softmax is applied
        aindex = one-hot encoding from memory
        self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=aindex), name="invloss")
        """
        # inverse_loss = self.cross_entropy(pred_acts, np.argmax(action_indices, dim=1))
        # inverse_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_acts, labels=action_indices))

        action_indices = tf.concat(action_indices, axis=0)
        cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
        inverse_loss = cross_entropy_loss(action_indices, pred_acts).numpy()
        return inverse_loss

    def value_loss(self, returns, values):
        """
        mean squared error between values as true labels and returns as predicted labels
        """
        return self.mse_loss(values, returns)

    def entropy(self, policy):
        """
        ensures there is some randomness or exploration in the actions
        entropy = - tf.reduce_mean(tf.reduce_sum(policy * log(policy), 1))
        """
        
        entropy = self.distribution(probs=policy).entropy()
        return entropy[0]

    def policy_loss(self, policy, advantage, action_indices=[]):
        """
        A2C policy loss calculation: -1/n * sum(advantage * log(policy)).
        """
        policy_logs = tf.math.log(tf.clip_by_value(policy, 1e-20, 1.0))

        # only take policies for taken actions
        pi_logs = tf.math.reduce_sum(tf.math.multiply(policy_logs, action_indices), axis=1)
        policy_loss = -tf.math.reduce_mean(advantage * pi_logs)

        return policy_loss

    def actor_critic_loss(self, policy, advantage, entropy):
        actor_loss, critic_loss = 0.0, 0.0
        alpha = 0.5
        beta = 0.001
        n = len(advantage)

        for policy, adv in zip(policy, advantage):
            actor_loss = actor_loss + (-tf.math.log(policy) * adv)
            critic_loss = critic_loss + (adv ** 2)
        
        actor_loss = actor_loss / n
        critic_loss = critic_loss / n
        total_loss = actor_loss + alpha * critic_loss + beta * entropy
        return total_loss
    
    def get_returns(self, memory):
        """
        Calculate expected value from rewards
        - At each timestep what was the total reward received after that timestep
        - Rewards in the past are discounted by multiplying them with gamma
        - These are the labels for our critic
        """

        """
        gamma = 0.99
        
        if memory.dones[-1]:  # game has terminated
            discounted_reward_sum = 0.
        else:  # bootstrap starting reward from last state
            discounted_reward_sum = memory.values[-1]
        
        returns = []
        for i in range(len(memory.rewards)-1, -1, -1):  # reverse buffer r
            reward = memory.rewards[i]
            if not memory.dones[i]:
                discounted_reward_sum = reward + gamma * discounted_reward_sum
            else:
                discounted_reward_sum = reward
            returns.append(discounted_reward_sum)
        
        returns.reverse()
        returns = np.array(returns)
        """
        discounted_reward_sum = 0
        returns = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            discounted_reward_sum = reward + gamma * discounted_reward_sum
            returns.append(discounted_reward_sum)
        returns.reverse()

        return returns

    def get_advantage(self, returns, values):
        advantage = []
        for ret, val in zip(returns, values):
            advantage.append(ret - val)
        
        # advantage = np.array(advantage)
        # advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-6)
        return advantage

    def compute_loss(self, memory, episode_reward, entropy):
        returns  = self.get_returns(memory)
        advantage = self.get_advantage(returns, memory.values)

        # policy_acts, new_value = self.act(np.stack(memory.frames))
        predicted_states = self.forward_act(
            tf.concat(memory.state_features, axis=0), 
            tf.concat(memory.action_indices, axis=0), 
            training=True)
        predicted_acts = self.inverse_act(
            tf.concat(memory.state_features, axis=0), 
            tf.concat(memory.new_state_features, axis=0), 
            training=True)
        
        """
        policy_loss = self.policy_loss(policy_acts, advantage, memory.action_indices)
        value_loss = self.value_loss(returns, new_value)
        entropy = self.entropy(policy_acts)
        loss = policy_loss + self.value_coeff * value_loss - self.ent_coeff * entropy
        """

        loss = self.actor_critic_loss(memory.policy, advantage, entropy)

        forward_loss = self.forward_loss(memory.new_state_features, predicted_states)
        inverse_loss = self.inverse_loss(predicted_acts, tf.concat(memory.action_indices, axis=0))

        agent_loss = (
            self.isc_lambda * loss
            + (1 - self.beta) * inverse_loss
            + self.beta * forward_loss
        )

        print("Loss Values:\nAC Loss = {}\nEntropy = {}\nForward Loss = {}\nInverse Loss = {}\nAgent Loss = {}\n".format(
            loss, entropy, forward_loss, inverse_loss, agent_loss))

        return agent_loss, forward_loss, inverse_loss
