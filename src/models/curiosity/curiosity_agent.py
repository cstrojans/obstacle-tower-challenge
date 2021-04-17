import datetime
import gym
import multiprocessing
import numpy as np
from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import os
from prettyprinter import pprint
import tensorboard
import tensorflow as tf
import tensorflow_probability as tfp
import time

from models.curiosity.agent import TowerAgent
from models.common.util import ActionSpace, CuriosityMemory
from models.common.util import record, instantiate_environment

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class CuriosityAgent():
    def __init__(self, env_path, train, evaluate, lr, timesteps, batch_size, gamma, save_dir, eval_seeds=[]):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.action_size = 8
        self._action_lookup = {
            0: np.asarray([0, 0, 0, 0]),  # no-op
            1: np.asarray([1, 0, 0, 0]),  # forward
            2: np.asarray([2, 0, 0, 0]),  # backward
            3: np.asarray([0, 1, 0, 0]),  # cam left
            4: np.asarray([0, 2, 0, 0]),  # cam right
            5: np.asarray([1, 0, 1, 0]),  # forward + jump
            6: np.asarray([1, 1, 0, 0]),  # forward + cam left
            7: np.asarray([1, 2, 0, 0]),  # forward + cam right
        }
        self.env_path = env_path
        self.env = instantiate_environment(env_path, train, evaluate, eval_seeds)
        self.input_shape = self.env.observation_space[0].shape  # (84, 84, 3)

        # model parameters
        self.agent = TowerAgent(self.action_size, self.input_shape)
        self.model_path = os.path.join(self.save_dir, 'model_curiosity')
        self.lr = lr
        self.gamma = gamma
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr, decay_steps=10000, decay_rate=0.9)
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, epsilon=1e-04)
        self.eps = np.finfo(np.float32).eps.item()  # smallest number such that 1.0 + eps != 1.0
        self._last_health = 99999.
        self._last_keys = 0
        self._last_floor = 0
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.ext_coeff = 1
        self.int_coeff = 1

        # logging
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './logs/curiosity/' + self.current_time
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)

    # def build_graph(self):
    #     """ build the model architecture """
    #     x = keras.Input(shape=self.input_shape)
    #     model = keras.Model(inputs=[x], outputs=self.agent.act(x, training=True))
    #     keras.utils.plot_model(model, to_file=os.path.join(
    #         self.save_dir, 'model_curiosity_architecture.png'), dpi=96, show_shapes=True, show_layer_names=True, expand_nested=False)
    #     return model

    def log_metrics(self, episode_reward, mean_reward, floor, ac_loss, forward_loss, inverse_loss, icm_loss, episode):
        with self.summary_writer.as_default():
            with tf.name_scope('curiosity'):
                tf.summary.scalar('episode_reward', episode_reward, step=episode)
                tf.summary.scalar('mean_reward', mean_reward, step=episode)
                tf.summary.scalar('floor', floor, step=episode)
                tf.summary.scalar('actor_critic_loss', ac_loss, step=episode)
                tf.summary.scalar('forward_model_loss', forward_loss, step=episode)
                tf.summary.scalar('inverse_model_loss', inverse_loss, step=episode)
                tf.summary.scalar('icm_loss', icm_loss, step=episode)
            self.summary_writer.flush()
    
    def load_model(self):
        print('Loading model from: {}'.format(self.model_path))
        self.agent.actor_critic_model = tf.keras.models.load_model(os.path.join(self.model_path, 'ac_model'), compile=False)
        self.agent.feature_extractor = tf.keras.models.load_model(os.path.join(self.model_path, 'fe_model'), compile=False)
        self.agent.forward_model = tf.keras.models.load_model(os.path.join(self.model_path, 'fm_model'), compile=False)
        self.agent.inverse_model = tf.keras.models.load_model(os.path.join(self.model_path, 'im_model'), compile=False)
    
    def save_model(self):
        print('Saving model to: {}'.format(self.model_path))
        tf.keras.models.save_model(self.agent.actor_critic_model, os.path.join(self.model_path, 'ac_model'))
        tf.keras.models.save_model(self.agent.feature_extractor, os.path.join(self.model_path, 'fe_model'))
        tf.keras.models.save_model(self.agent.forward_model, os.path.join(self.model_path, 'fm_model'))
        tf.keras.models.save_model(self.agent.inverse_model, os.path.join(self.model_path, 'im_model'))

    def update(self, tape, ac_loss, forward_loss, inverse_loss, icm_loss):
        """ calculate and apply gradients """
        ac_grads = tape.gradient(ac_loss, self.agent.actor_critic_model.trainable_variables)
        fe_grads = tape.gradient(icm_loss, self.agent.feature_extractor.trainable_variables)
        fm_grads = tape.gradient(forward_loss, self.agent.forward_model.trainable_variables)
        im_grads = tape.gradient(inverse_loss, self.agent.inverse_model.trainable_variables)
        
        self.opt.apply_gradients(zip(ac_grads, self.agent.actor_critic_model.trainable_variables))
        self.opt.apply_gradients(zip(fe_grads, self.agent.feature_extractor.trainable_variables))
        self.opt.apply_gradients(zip(fm_grads, self.agent.forward_model.trainable_variables))
        self.opt.apply_gradients(zip(im_grads, self.agent.inverse_model.trainable_variables))
    
    
    def get_updated_reward(self, reward, new_health, new_keys, done):
        new_health = float(new_health)
        new_reward = 0.0
        # opened a door, solved a puzzle, picked up a key
        if 0.1 <= reward < 1:
            new_reward += 0.5
        
        # crossing a floor - between [1, 4]
        if reward >= 1:
            new_reward += (new_health / 10000)
        
        # found time orb / crossed a floor
        if new_health > self._last_health:
            new_reward += 0.5
        
        return new_reward

    def train(self):
        """ train the model """
        start_time = time.time()

        self.agent.restore_checkpoint()
        mem = CuriosityMemory()
        running_reward = 0.
        best_score = 0.
        done = False
        obs = self.env.reset()
        state, _, _, _ = obs

        agent_loss, feature_extractor_loss, forward_loss, inverse_loss, loss = 0.0, 0.0, 0.0, 0.0, 0.0
        entropy_term = 0.0
        episode_reward = 0.0
        extrinsic_reward = 0.0
        episode_steps = 0
        episode = 0
        timestep = 0

        while timestep <= self.timesteps:
            i = 0
            with tf.GradientTape(persistent=True) as tape:
                while i < self.batch_size:
                    # collect experience
                    # get action as per policy
                    exp_state = tf.convert_to_tensor(state)
                    exp_state = tf.expand_dims(exp_state, axis=0)
                    policy, value = self.agent.act(exp_state, training=True)
                    
                    entropy = -np.sum(policy * np.log(policy))
                    entropy_term += entropy
                    
                    # choose most probable action
                    dist = tfp.distributions.Categorical(probs=policy, dtype=tf.float32)
                    action_index = int(dist.sample().numpy())
                    action = self._action_lookup[action_index]

                    action_one_hot = tf.one_hot(action_index, self.action_size)
                    action_one_hot = np.reshape(action_one_hot, (1, self.action_size))
                    
                    # perform action in game env
                    frame_reward = 0.0
                    for _ in range(4):  # frame skipping
                        obs, reward, done, _ = self.env.step(action)
                        new_state, new_keys, new_health, cur_floor = obs
                        reward = self.get_updated_reward(reward, new_health, new_keys, done)
                        frame_reward += reward
                        self._last_health = new_health
                        self._last_keys = new_keys
                        if cur_floor > self._last_floor:
                            self._last_floor = cur_floor
                        i += 1
                        episode_steps += 1
                        timestep += 1
                    
                    intrinsic_reward, state_features, new_state_features = self.agent.icm_act(state, new_state, action_one_hot, training=True)
                    # total_reward = self.ext_coeff * reward + self.int_coeff * intrinsic_reward
                    total_reward = frame_reward + intrinsic_reward
                    extrinsic_reward += frame_reward
                    episode_reward += total_reward
                    
                    # store experience
                    mem.store(new_state,
                                total_reward,
                                done,
                                value[0, 0],
                                action_one_hot,
                                tf.math.log(policy[0, action_index]),
                                state_features, # (1, 288)
                                new_state_features)
                    
                    if done:
                        break
                
                ac_loss, forward_loss, inverse_loss, icm_loss = self.agent.compute_loss(mem, entropy_term)
            
            self.update(tape, ac_loss, forward_loss, inverse_loss, icm_loss)
            mem.clear()

            if done:  # reset parameters and print episode statistics
                running_reward = (running_reward * episode + extrinsic_reward) / (episode + 1)
                episode += 1
                self.log_metrics(extrinsic_reward, running_reward, self._last_floor, ac_loss, forward_loss, inverse_loss, icm_loss, episode)
                print("Episode: {} | Episode Reward: {:.3f} | Mean Reward: {:.3f} | AC Loss: {:.3f} | FM Loss: {:.3f} | IM Loss: {:.3f} | ICM Loss: {:.3f} | Floor: {} | Steps: {} | Total Steps: {}".format(
                    episode, extrinsic_reward, running_reward, ac_loss, forward_loss, inverse_loss, icm_loss, self._last_floor, episode_steps, timestep))
                obs = self.env.reset()
                state, _, _, _ = obs

                if extrinsic_reward > best_score:
                    print("Found better score: old = {}, new = {}".format(best_score, extrinsic_reward))
                    best_score = extrinsic_reward
                    self.agent.save_checkpoint()
                    self.save_model()
                
                self._last_health = 99999.
                self._last_keys = 0
                self._last_floor = 0
                episode_reward = 0.0
                extrinsic_reward = 0.0
                episode_steps = 0
                agent_loss, forward_loss, inverse_loss = 0.0, 0.0, 0.0
                entropy_term = 0.0

        self.agent.save_checkpoint()
        self.save_model()
        end_time = time.time()
        print("\nTraining complete. Time taken = {} secs".format(end_time - start_time))
        self.env.close()

    def play_single_episode(self):
        """ have the trained agent play a single game """
        action_space = ActionSpace()
        self.load_model()
        print("Playing single episode...")
        done = False
        step_counter = 0
        reward_sum = 0
        obs = self.env.reset()
        state, _, _, _ = obs

        try:
            while not done:
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, axis=0)
                policy, _ = self.agent.act(state)
                action_index = np.random.choice(self.action_size, p=np.squeeze(policy))
                action = self._action_lookup[action_index]
                
                for i in range(4):  # frame skipping
                    obs, reward, done, _ = self.env.step(action)
                    state, _, _, _ = obs
                    reward_sum += reward
                    step_counter += 1
                
                print("{}. Reward: {}, action: {}".format(step_counter,
                      reward_sum, action_space.get_action_meaning(action)))
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        except Exception as e:
            print(str(e))
        finally:
            if not self.evaluate:
                self.env.close()

        print("Game play completed.")
        return reward_sum

    def evaluate(self):
        """ run episodes until evaluation is complete """
        while not self.env.evaluation_complete:
            episode_reward = self.play_single_episode()

        pprint(self.env.results)
        self.env.close()
