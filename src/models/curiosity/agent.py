import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Model, layers, activations, losses
from tensorflow.keras.layers import Layer
from models.curiosity.networks import ConvGruNet, FeatureExtractor, ForwardModel, InverseModel


class TowerAgent(object):
    """ Agent that learns using intrinsic curiosity model (ICM) """
    def __init__(self, action_size=7, ip_shape=(84, 84, 3)):
        super(TowerAgent, self).__init__()
        self.action_size = action_size
        self.ip_shape = ip_shape

        self.actor_critic_model = ConvGruNet(self.action_size, self.ip_shape)
        self.feature_extractor = FeatureExtractor(self.ip_shape)
        self.forward_model = ForwardModel()
        self.inverse_model = InverseModel(self.action_size)
        
        self.gamma = 0.99
        self.ent_coeff = 0.01
        self.eta = 0.1
        self.value_coeff = 0.5
        self.beta = 0.5  # weighs the inverse model loss against the forward model loss
        self.isc_lambda = 0.8  # weighs the importance of the policy gradient loss against the intrinsic reward signal

        # checkpointing
        self.ac_ckpt = tf.train.Checkpoint(model=self.actor_critic_model, step=tf.Variable(1))
        self.fe_ckpt = tf.train.Checkpoint(model=self.feature_extractor, step=tf.Variable(1))
        self.fm_ckpt = tf.train.Checkpoint(model=self.forward_model, step=tf.Variable(1))
        self.im_ckpt = tf.train.Checkpoint(model=self.inverse_model, step=tf.Variable(1))
        
        self.ac_manager = tf.train.CheckpointManager(self.ac_ckpt, directory='./checkpoints/model-curiosity/ac_ckpts', max_to_keep=3)
        self.fe_manager = tf.train.CheckpointManager(self.fe_ckpt, directory='./checkpoints/model-curiosity/fe_ckpts', max_to_keep=3)
        self.fm_manager = tf.train.CheckpointManager(self.fm_ckpt, directory='./checkpoints/model-curiosity/fm_ckpts', max_to_keep=3)
        self.im_manager = tf.train.CheckpointManager(self.im_ckpt, directory='./checkpoints/model-curiosity/im_ckpts', max_to_keep=3)
    
    def save_checkpoint(self):
        self.ac_manager.save()
        self.fe_manager.save()
        self.fm_manager.save()
        self.im_manager.save()

        print("Saved checkpoint for step {}".format(int(self.ac_ckpt.step)))
        self.ac_ckpt.step.assign_add(1)
        self.fe_ckpt.step.assign_add(1)
        self.fm_ckpt.step.assign_add(1)
        self.im_ckpt.step.assign_add(1)

    def restore_checkpoint(self):
        if self.ac_manager.latest_checkpoint:
            self.ac_ckpt.restore(self.ac_manager.latest_checkpoint)
            self.fe_ckpt.restore(self.fe_manager.latest_checkpoint)
            self.fm_ckpt.restore(self.fm_manager.latest_checkpoint)
            self.im_ckpt.restore(self.im_manager.latest_checkpoint)
            print("Restored from {}".format(self.ac_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    def mse_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
        # return tf.keras.losses.MSE(y_true, y_pred).numpy()
        
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
        pred_state_features = self.forward_model((state_features, action_one_hot))

        prediction_error = 0.5 * tf.square(new_state_features - pred_state_features)
        prediction_error = tf.reduce_mean(prediction_error) * 3136
        intrinsic_reward = (self.eta / 2) * prediction_error
        intrinsic_reward = tf.clip_by_value(intrinsic_reward, -0.1, 0.1)

        return intrinsic_reward, state_features, new_state_features

    def forward_act(self, batch_state_features, batch_action_indices):
        """ predicts the features of the next state """
        batch_state_features = tf.cast(batch_state_features, tf.float32)
        batch_action_indices = tf.cast(batch_action_indices, tf.float32)
        return self.forward_model((batch_state_features, batch_action_indices))

    def inverse_act(self, batch_state_features, batch_new_state_features):
        """ predicts the action performed """
        batch_state_features = tf.cast(batch_state_features, tf.float32)
        batch_new_state_features = tf.cast(batch_new_state_features, tf.float32)
        return self.inverse_model((batch_state_features, batch_new_state_features))

    def forward_loss(self, new_state_features, new_state_pred):
        """ prediction error between phi(s_t+1) and phi(s'_t+1) """
        forward_loss = self.mse_loss(new_state_features, new_state_pred)
        return forward_loss

    def inverse_loss(self, pred_acts, action_indices):
        """
        logits = output of inverse model - before softmax is applied
        aindex = one-hot encoding from memory
        """
        action_indices = tf.concat(action_indices, axis=0)
        inverse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_acts, labels=action_indices))
        return inverse_loss

    def actor_critic_loss(self, policy, values, returns, entropy):
        advantage = tf.math.subtract(returns, values)
        # actor_loss = tf.math.reduce_mean(tf.clip_by_value(advantage * policy, -20.0, 20.0))
        actor_loss = tf.math.reduce_mean(tf.stop_gradient(advantage) * policy)
        critic_loss = tf.keras.losses.MSE(returns, values).numpy()

        total_loss = actor_loss + self.value_coeff * critic_loss + self.ent_coeff * entropy
        return total_loss  # negate it to perform gradient ascent
        
    def get_returns(self, rewards, last_done, last_value):
        """
        calculate expected value from rewards. They are labels for the critic
        """
        if last_done:  # game has terminated
            discounted_reward_sum = 0.
        else:  # bootstrap starting reward from last state
            discounted_reward_sum = last_value
        returns = []

        for reward in rewards[::-1]:  # reverse buffer r
            discounted_reward_sum = reward + self.gamma * discounted_reward_sum
            returns.append(discounted_reward_sum)
        returns.reverse()

        return returns
    
    def compute_loss(self, memory, entropy):
        returns  = self.get_returns(memory.rewards, memory.dones[-1], memory.values[-1])
        
        policy_acts, new_value = self.act(np.stack(memory.frames))
        predicted_states = self.forward_act(
            tf.concat(memory.state_features, axis=0), 
            tf.concat(memory.action_indices, axis=0))
        predicted_acts = self.inverse_act(
            tf.concat(memory.state_features, axis=0), 
            tf.concat(memory.new_state_features, axis=0))

        ac_loss = self.actor_critic_loss(memory.policy, memory.values, returns, entropy)
        forward_loss = self.forward_loss(memory.new_state_features, predicted_states)
        inverse_loss = self.inverse_loss(predicted_acts, memory.action_indices)
        model_loss = self.beta * forward_loss + (1 - self.beta) * inverse_loss

        return ac_loss, forward_loss, inverse_loss, model_loss
