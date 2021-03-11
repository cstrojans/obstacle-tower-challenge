import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, layers, activations, losses
from tensorflow.keras.layers import Layer
from models.architecture.curiosity_networks import ConvGruNet, FeatureExtractor, ForwardModel, InverseModel, ValueNetwork, PolicyNetwork


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

        self.mse_loss = losses.MSE()
        self.ent_coeff = 0.001
        self.value_coeff = 0.5
        self.beta = 0.8
        self.isc_lambda = 0.8
    
    def act(self, state):
        """ get estimated policy and value """
        state = tf.expand_dims(state, axis=0)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        policy, value = self.actor_critic_model(state)

        return policy, value

    def icm_act(self, state, new_state, action_indices, eta=0.1):
        """ get intrinsic reward 
        Intrinsic reward calculation:
            eta/2 * mean((F'(St+1) - F(St+1))^2)
        """
        state_features = self.feature_extractor(state)
        new_state_features = self.feature_extractor(new_state)
        pred_state_features = self.forward_model(state_features, action_indices)

        intrinsic_reward = (eta / 2) * self.mse_loss(pred_state_features, new_state_features)
        return intrinsic_reward, state_features, new_state_features

    def forward_act(self, batch_state_features, batch_action_indices):
        return self.forward_model(batch_state_features, batch_action_indices)

    def inverse_act(self, batch_state_features, batch_new_state_features):
        return self.inverse_model(batch_state_features, batch_new_state_features)

    def forward_loss(self, new_state_features, new_state_pred):
        """
        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
        """
        forward_loss = 0.5 * self.mse_loss(new_state_pred, new_state_features)
        return forward_loss

    def inverse_loss(self, pred_acts, action_indices):
        """
        logits = output of inverse model - before softmax is applied
        aindex = one-hot encoding from memory
        self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=aindex), name="invloss")
        """
        inverse_loss = self.cross_entropy(pred_acts, np.argmax(action_indices, dim=1))
        return inverse_loss

    def value_loss(self, returns, values):
        """
        vf_loss = 0.5 * tf.reduce_mean(tf.square(pi.vf - self.r))
        """
        return 0.5 * self.mse_loss(values, returns)

    def entropy(self, policy):
        """
        ensures there is some randomness or exploration in the actions
        entropy = - tf.reduce_mean(tf.reduce_sum(prob_tf * log_prob_tf, 1))
        """
        dist = tfp.distributions.Categorical
        return dist(probs=policy).entropy()

    def policy_loss(self, policy, advantage, action_indices):
        """
        A2C policy loss calculation: -1/n * sum(advantage * log(policy)).

        pi_loss = - tf.reduce_mean(tf.reduce_sum(log_prob_tf * self.ac, 1) * self.adv)
        """

        policy_logs = tf.math.log(tf.clip_by_value(policy, 1e-20, 1.0))

        # only take policies for taken actions
        pi_logs = tf.math.reduce_sum(tf.math.multiply(policy_logs, action_indices), axis=1)
        policy_loss = -tf.math.reduce_mean(advantage * pi_logs)

        return policy_loss

    def get_returns(self, memory):
        """
        Calculate expected value from rewards
        - At each timestep what was the total reward received after that timestep
        - Rewards in the past are discounted by multiplying them with gamma
        - These are the labels for our critic
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
        return returns

    def get_advantage(self, memory, returns):
        advantage = []
        for i in range(len(returns)):
            advantage.append(returns[i] - memory.values[i])
        
        advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-6)
        return advantage

    def compute_loss(self, memory, episode_reward):
        returns  = self.get_returns(memory)
        advantage = self.get_advantage(memory, returns)

        policy_acts, new_value = self.agent_network.act(memory.frames)

        predicted_states = self.agent_network.forward_act(
            memory.state_features, memory.action_indices
        )
        predicted_acts = self.agent_network.inverse_act(
            memory.state_features, memory.new_state_features
        )

        
        policy_loss = self.policy_loss(policy_acts, advantage, memory.action_indices)
        value_loss = self.value_loss(returns, memory.values)
        entropy = self.entropy(policy_acts)

        loss = policy_loss + self.value_coeff * value_loss - self.ent_coeff * entropy
        forward_loss = self.forward_loss(memory.new_state_features, predicted_states)
        inverse_loss = self.inverse_loss(predicted_acts, memory.action_indices)

        agent_loss = (
            self.isc_lambda * loss
            + (1 - self.beta) * inverse_loss
            + self.beta * forward_loss
        )

        return agent_loss, forward_loss, inverse_loss
    
    def update(self, global_model, tape, agent_loss, forward_loss, inverse_loss):
        # calculate local gradients
        ac_grads = tape.gradient(agent_loss, self.actor_critic_model.trainable_variables)
        fe_grads = tape.gradient(agent_loss, self.feature_extractor.trainable_variables)
        forward_grads = tape.gradient(forward_loss, self.forward_model.trainable_variables)
        inverse_grads = tape.gradient(inverse_loss, self.inverse_model.trainable_variables)
        
        # send local gradients to global model
        self.opt.apply_gradients(zip(ac_grads, global_model.actor_critic_model.trainable_variables))
        self.opt.apply_gradients(zip(fe_grads, global_model.feature_extractor.trainable_variables))
        self.opt.apply_gradients(zip(forward_grads, global_model.forward_model.trainable_variables))
        self.opt.apply_gradients(zip(inverse_grads, global_model.inverse_model.trainable_variables))

        # update local model with new weights
        self.local_model.set_weights(self.global_model.get_weights())
        self.local_model.actor_critic_model.set_weights(self.global_model.actor_critic_model.get_weights())
        self.local_model.feature_extractor.set_weights(self.global_model.feature_extractor.get_weights())
        self.local_model.forward_model.set_weights(self.global_model.forward_model.get_weights())
        self.local_model.inverse_model.set_weights(self.global_model.inverse_model.get_weights())
            