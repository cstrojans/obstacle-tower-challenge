import datetime
import gym
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import os
from prettyprinter import pprint
from queue import Queue
import tensorboard
import tensorflow as tf
from tensorflow import keras
import threading
import time

from models.curiosity.agent import TowerAgent
from models.common.util import ActionSpace, CuriosityMemory
from models.common.util import record, instantiate_environment

matplotlib.use('agg')
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class CuriosityAgent():
    def __init__(self, env_path, train, evaluate, lr, timesteps, batch_size, gamma, save_dir, eval_seeds=[]):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.action_size = 7
        self._action_lookup = {
            0: np.asarray([1, 0, 0, 0]),  # forward
            1: np.asarray([2, 0, 0, 0]),  # backward
            2: np.asarray([0, 1, 0, 0]),  # cam left
            3: np.asarray([0, 2, 0, 0]),  # cam right
            4: np.asarray([1, 0, 1, 0]),  # jump forward
            5: np.asarray([1, 1, 0, 0]),  # forward + cam left
            6: np.asarray([1, 2, 0, 0]),  # forward + cam right
        }
        self.env_path = env_path
        self.env = instantiate_environment(env_path, train, evaluate, eval_seeds)
        self.input_shape = self.env.observation_space[0].shape  # (84, 84, 3)

        # model parameters
        self.agent = TowerAgent(self.action_size, self.input_shape)
        self.model_path = os.path.join(self.save_dir, 'model_curiosity')
        self.lr = lr
        self.gamma = gamma
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr, decay_steps=1000, decay_rate=0.9)
        self.opt = keras.optimizers.Adam(learning_rate=self.lr_schedule, epsilon=1e-05)
        self.eps = np.finfo(np.float32).eps.item()  # smallest number such that 1.0 + eps != 1.0
        self._last_health = 99999.
        self._last_keys = 0
        self.timesteps = timesteps
        self.batch_size = batch_size

        # logging
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './logs/' + self.current_time + '/curiosity'
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)

    # def build_graph(self):
    #     """ build the model architecture """
    #     x = keras.Input(shape=self.input_shape)
    #     model = keras.Model(inputs=[x], outputs=self.agent.act(x, training=True))
    #     keras.utils.plot_model(model, to_file=os.path.join(
    #         self.save_dir, 'model_curiosity_architecture.png'), dpi=96, show_shapes=True, show_layer_names=True, expand_nested=False)
    #     return model

    def log_metrics(self, episode_reward, running_reward, ac_loss, forward_loss, inverse_loss, agent_loss, episode):
        with self.summary_writer.as_default():
            tf.summary.scalar('episode_reward', episode_reward, step=episode)
            tf.summary.scalar('moving_average_reward', running_reward, step=episode)
            tf.summary.scalar('actor_critic_loss', ac_loss, step=episode)
            tf.summary.scalar('forward_model_loss', forward_loss, step=episode)
            tf.summary.scalar('inverse_model_loss', inverse_loss, step=episode)
            tf.summary.scalar('agent_loss', agent_loss, step=episode)
            self.summary_writer.flush()
    
    def load_model(self):
        print('\nLoading model from: {}\n'.format(self.model_path))
        self.agent.actor_critic_model = keras.models.load_model(os.path.join(self.model_path, 'ac_model'), compile=True)
        self.agent.feature_extractor = keras.models.load_model(os.path.join(self.model_path, 'fe_model'), compile=True)
        self.agent.forward_model = keras.models.load_model(os.path.join(self.model_path, 'fm_model'), compile=True)
        self.agent.inverse_model = keras.models.load_model(os.path.join(self.model_path, 'im_model'), compile=True)
    
    def save_model(self):
        print('\nSaving model to: {}\n'.format(self.model_path))
        keras.models.save_model(self.agent.actor_critic_model, os.path.join(self.model_path, 'ac_model'))
        keras.models.save_model(self.agent.feature_extractor, os.path.join(self.model_path, 'fe_model'))
        keras.models.save_model(self.agent.forward_model, os.path.join(self.model_path, 'fm_model'))
        keras.models.save_model(self.agent.inverse_model, os.path.join(self.model_path, 'im_model'))

    def update(self, tape, ac_loss, agent_loss, forward_loss, inverse_loss):
        """ calculate and apply gradients """
        ac_grads = tape.gradient(ac_loss, self.agent.actor_critic_model.trainable_variables)
        fe_grads = tape.gradient(ac_loss, self.agent.feature_extractor.trainable_variables)
        fm_grads = tape.gradient(forward_loss, self.agent.forward_model.trainable_variables)
        im_grads = tape.gradient(inverse_loss, self.agent.inverse_model.trainable_variables)
        
        self.opt.apply_gradients(zip(ac_grads, self.agent.actor_critic_model.trainable_variables))
        self.opt.apply_gradients(zip(fe_grads, self.agent.feature_extractor.trainable_variables))
        self.opt.apply_gradients(zip(fm_grads, self.agent.forward_model.trainable_variables))
        self.opt.apply_gradients(zip(im_grads, self.agent.inverse_model.trainable_variables))
    
    def train(self):
        """ train the model """
        start_time = time.time()
        
        mem = CuriosityMemory()
        rewards = []
        running_reward = 0.
        best_score = 0.
        global_steps = 0
        num_games = 0
        num_updates = self.timesteps // self.batch_size
        done = True
        print("Number of updates = {}".format(num_updates))

        for timestamp in range(1, num_updates+1):
            agent_loss, forward_loss, inverse_loss = 0.0, 0.0, 0.0
            entropy_term = 0.0

            if done:
                episode_reward = 0.0
                episode_steps = 0
                num_games += 1
                obs = self.env.reset()
                state, _, _, _ = obs
                done = False

            # collect experience
            with tf.GradientTape(persistent=True) as tape:
                for episode_step in range(self.batch_size):
                    # get action as per policy
                    exp_state = tf.convert_to_tensor(state)
                    exp_state = tf.expand_dims(exp_state, axis=0)
                    policy, value = self.agent.act(exp_state, training=True)
                    
                    entropy = -np.sum(policy * np.log(policy))
                    entropy_term += entropy
                    
                    # choose most probable action
                    action_index = np.random.choice(self.action_size, p=np.squeeze(policy))
                    action = self._action_lookup[action_index]
                    action_one_hot = np.zeros(self.action_size, dtype=np.float32)
                    action_one_hot[action_index] = 1
                    action_one_hot = np.reshape(action_one_hot, (1, self.action_size))

                    # perform action in game env
                    # TODO: implement frame skipping
                    obs, reward, done, _ = self.env.step(action)
                    new_state, new_keys, new_health, cur_floor = obs
                    
                    intrinsic_reward, state_features, new_state_features = self.agent.icm_act(state, new_state, action_one_hot, training=True)
                    total_reward = reward + intrinsic_reward
                    episode_reward += total_reward
                    episode_steps += 1
                    global_steps += 1

                    # store experience
                    mem.store(new_state,
                              total_reward,
                              done,
                              value[0, 0],
                              action_one_hot,
                              policy[0, action_index],
                              state_features, # (1, 288)
                              new_state_features)

                    if done:  # game has terminated
                        break
                
                ac_loss, agent_loss, forward_loss, inverse_loss = self.agent.compute_loss(mem, episode_reward, entropy_term)
                self.update(tape, ac_loss, agent_loss, forward_loss, inverse_loss)

            # clear the experience
            mem.clear()

            if done:
                rewards.append(episode_reward)
                running_reward = sum(rewards[-10:]) / 10
                self.log_metrics(episode_reward, running_reward, ac_loss, forward_loss, inverse_loss, agent_loss, timestamp)
                print("Episode: {} | Average Reward: {:.3f} | Episode Reward: {:.3f} | AC Loss: {:.3f} | FM Loss: {:.3f} | IM Loss: {:.3f} | Steps: {} | Total Steps: {}".format(
                    num_games, running_reward, episode_reward, ac_loss, forward_loss, inverse_loss, episode_steps, global_steps))

                if episode_reward > best_score:
                    print("\nFound better score: old = {}, new = {}\n".format(best_score, episode_reward))
                    best_score = episode_reward
                    self.save_model()

        end_time = time.time()
        print("\nTraining complete. Time taken = {} secs".format(end_time - start_time))
        self.env.close()

    def play_single_episode(self):
        """ have the trained agent play a single game """
        action_space = ActionSpace()
        model = self.load_model()
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
                policy, _ = model.act(state, training=False)
                action_index = np.random.choice(self.action_size, p=np.squeeze(policy))
                action = self._action_lookup[action_index]
                
                for i in range(4):  # frame skipping
                    obs, reward, done, _ = self.env.step(action)
                    state, _, _, _ = obs
                    reward_sum += reward
                    step_counter += 1
                
                if not self.evaluate:
                    print("{}. Reward: {}, action: {}".format(
                        step_counter, reward_sum, action_space.get_action_meaning(action)))
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            if not self.evaluate:
                self.env.close()
                print("Environment closed.")
            print("Game play completed.")
            return reward_sum

    def evaluate(self):
        """ run episodes until evaluation is complete """
        while not self.env.evaluation_complete:
            episode_reward = self.play_single_episode()

        pprint(self.env.results)
        self.env.close()
