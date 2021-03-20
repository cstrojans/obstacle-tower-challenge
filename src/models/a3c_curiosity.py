import gym
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import os
from prettyprinter import pprint
from queue import Queue
import tensorflow as tf
from tensorflow import keras
import threading
import time

from models.architecture.curiosity import TowerAgent
from models.util import ActionSpace, Memory, CuriosityMemory
from models.util import record
from definitions import *

matplotlib.use('agg')
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class CuriosityMasterAgent():
    """MasterAgent A3C: Curiosity Model
    """

    def __init__(self, env_path, train, evaluate, lr, max_eps, update_freq, gamma, num_workers, save_dir, eval_seeds=[]):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.env_path = env_path
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
        self.num_workers = multiprocessing.cpu_count() if num_workers == 0 else num_workers
        if train:
            self.env = ObstacleTowerEnv(
                env_path, worker_id=0, retro=False, realtime_mode=False, greyscale=False, config=train_env_reset_config)
        else:
            if evaluate:
                self.env = ObstacleTowerEnv(
                    env_path, worker_id=0, retro=False, realtime_mode=False, greyscale=False, config=eval_env_reset_config)
                self.env = ObstacleTowerEvaluation(self.env, eval_seeds)
            else:
                self.env = ObstacleTowerEnv(
                    env_path, worker_id=0, retro=False, realtime_mode=True, greyscale=False, config=eval_env_reset_config)
        self.input_shape = self.env.observation_space[0].shape  # (84, 84, 3)

        # model parameters
        self.lr = lr
        self.max_eps = max_eps
        self.update_freq = update_freq
        self.gamma = gamma
        self.model_path = os.path.join(self.save_dir, 'model_a3c')

        self.global_agent = TowerAgent(self.action_size, self.input_shape)

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr, decay_steps=1000, decay_rate=0.9)
        self.opt = keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-05)

        vec = np.random.random(self.input_shape)
        vec = np.expand_dims(vec, axis=0)
        vec = tf.convert_to_tensor(vec, dtype=tf.float32)
        self.global_agent.act(vec, training=True)

    def build_graph(self):
        """ build the model architecture """
        x = keras.Input(shape=self.input_shape)
        # TODO: replace call() with something else to get output tensor representation
        model = keras.Model(inputs=[x], outputs=self.global_agent.act(x, training=True))
        keras.utils.plot_model(model, to_file=os.path.join(
            self.save_dir, 'model_curiosity_architecture.png'), dpi=96, show_shapes=True, show_layer_names=True, expand_nested=False)
        return model

    def train(self):
        """ instantiate multiple workers and train the global model """
        start_time = time.time()
        res_queue = Queue()
        workers = [Worker(self.action_size,
                          self._action_lookup,
                          self.global_agent,
                          self.opt,
                          self.max_eps,
                          self.update_freq,
                          self.gamma,
                          res_queue,
                          worker_id,
                          self.env_path,
                          save_dir=self.save_dir) for worker_id in range(1, self.num_workers+1)]

        for i, worker in enumerate(workers, start=1):
            print("Starting worker {}".format(i))
            worker.start()

        # record episode reward to plot
        moving_average_rewards = []
        losses = []
        while True:
            result = res_queue.get()
            if result:
                reward, loss = result
            else:
                break
            if reward is None or not loss:
                break
            else:
                moving_average_rewards.append(reward)
                losses.append(loss)
        
        for w in workers:
            w.join()

        end_time = time.time()
        print("\nTraining complete. Time taken = {} secs".format(
            end_time - start_time))

        """
        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average episode reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(self.save_dir, 'model_curiosity_moving_average.png'))
        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        fig.suptitle('A3C Model')

        ax1.plot(range(1, len(moving_average_rewards) + 1), moving_average_rewards)
        ax1.set_title('Reward vs Timesteps')
        ax1.set(xlabel='Episodes', ylabel='Reward')

        ax2.plot(range(1, len(losses) + 1), losses)
        ax2.set_title('Loss vs Timesteps')
        ax2.set(xlabel='Episodes', ylabel='Loss')
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, 'model_curiosity_moving_average.png'))

        # save the trained model to a file
        # print('Saving global model to: {}'.format(self.model_path))
        # keras.models.save_model(self.global_agent, self.model_path)
        self.env.close()

    def play_single_episode(self):
        """ have the trained agent play a single game """
        action_space = ActionSpace()
        print('Loading model from: {}'.format(self.model_path))
        model = keras.models.load_model(self.model_path, compile=True)
        print("Playing single episode...")
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            obs = self.env.reset()
            state, _, _, _ = obs
            
            while not done:
                policy, _ = model.act(tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32), training=True)
                action_index = np.random.choice(self.action_size, p=np.squeeze(policy))
                action = self._action_lookup[action_index]
                
                for i in range(4):  # frame skipping
                    obs, reward, done, _ = self.env.step(action)
                    state, _, _, _ = obs
                    reward_sum += reward
                    step_counter += 1
                
                if done:
                    break
                
                print("{}. Reward: {}, action: {}".format(step_counter,
                                                          reward_sum, action_space.get_action_meaning(action)))
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


class Worker(threading.Thread):
    running_reward = 0.
    best_score = 0.
    global_steps = 0
    save_lock = threading.Lock()

    def __init__(self, action_size, action_lookup, global_agent, opt_fn, max_eps,
                 update_freq, gamma, result_queue, idx, env_path, save_dir):
        super(Worker, self).__init__()
        self.result_queue = result_queue
        self.worker_idx = idx
        self.save_dir = save_dir

        self.env = ObstacleTowerEnv(env_path, worker_id=self.worker_idx,
                                    retro=False, realtime_mode=False, greyscale=False, config=train_env_reset_config)
        self.action_size = action_size
        self._action_lookup = action_lookup
        self.input_shape = self.env.observation_space[0].shape  # (84, 84, 3)
        self._last_health = 99999.
        self._last_keys = 0

        self.global_agent = global_agent
        self.local_agent = TowerAgent(self.action_size, self.input_shape)
        self.opt = opt_fn
        self.max_eps = max_eps
        self.update_freq = update_freq
        self.gamma = gamma
        self.eps = np.finfo(np.float32).eps.item()  # smallest number such that 1.0 + eps != 1.0
        # self.timesteps = 5000000
        # self.batch_size = 128
        self.timesteps = 1024
        self.batch_size = 128
        self.model_path = os.path.join(self.save_dir, 'model_curiosity')

    def get_updated_reward(self, reward, new_health, new_keys, done):
        new_health = float(new_health)
        if done:  # penalize when game is terminated
            self._last_health = 99999.
            self._last_keys = 0
            reward = -1
        else:
            if reward >= 1:  # prioritize crossing a floor/level - between [1, 4]
                reward += (new_health / 10000)
            
            if new_health > self._last_health:  # found time orb
                reward += 1

            if new_keys > self._last_keys:  # found a key
                reward += 1

        return reward

    def update(self, tape, agent_loss, forward_loss, inverse_loss):
        # calculate local gradients
        variables = {
            'ac_vars': self.local_agent.actor_critic_model.trainable_variables,
            'fe_vars': self.local_agent.feature_extractor.trainable_variables,
            'fm_vars': self.local_agent.forward_model.trainable_variables,
            'im_vars': self.local_agent.inverse_model.trainable_variables
        }
        grads = tape.gradient(agent_loss, variables)
        
        # send local gradients to global model
        self.opt.apply_gradients(zip(grads['ac_vars'], self.global_agent.actor_critic_model.trainable_variables))
        self.opt.apply_gradients(zip(grads['fe_vars'], self.global_agent.feature_extractor.trainable_variables))
        self.opt.apply_gradients(zip(grads['fm_vars'], self.global_agent.forward_model.trainable_variables))
        self.opt.apply_gradients(zip(grads['im_vars'], self.global_agent.inverse_model.trainable_variables))

        # update local model with new weights
        self.local_agent.actor_critic_model.set_weights(self.global_agent.actor_critic_model.get_weights())
        self.local_agent.feature_extractor.set_weights(self.global_agent.feature_extractor.get_weights())
        self.local_agent.forward_model.set_weights(self.global_agent.forward_model.get_weights())
        self.local_agent.inverse_model.set_weights(self.global_agent.inverse_model.get_weights())
            
        print("Gradient calculation and update completed.")
    
    def run(self):
        mem = CuriosityMemory()
        rewards = []
        entropy_term = 0
        num_updates = self.timesteps // self.batch_size

        for timestamp in range(num_updates):
            reset = True
            episode_reward = 0.0
            episode_steps = 0
            agent_loss, forward_loss, inverse_loss = 0.0, 0.0, 0.0
            
            # collect experience
            with tf.GradientTape(persistent=True) as tape:
                for episode_step in range(self.batch_size):
                    if reset:
                        obs = self.env.reset()
                        state, _, _, _ = obs
                        reset = False
                    
                    exp_state = tf.convert_to_tensor(state, dtype=tf.float32)
                    exp_state = tf.expand_dims(exp_state, axis=0)
                    policy, value = self.local_agent.act(exp_state, training=True)
                    entropy = -np.sum(policy * np.log(policy))
                    entropy_term += entropy
                    
                    action_index = np.random.choice(self.action_size, p=np.squeeze(policy))
                    action_one_hot = np.zeros(self.action_size)
                    action_one_hot[action_index] = 1
                    action_one_hot = np.reshape(action_one_hot, (1, self.action_size))
                    action = self._action_lookup[action_index]

                    # TODO: implement frame skipping
                    obs, reward, done, _ = self.env.step(action)
                    new_state, new_keys, new_health, cur_floor = obs
                        
                    intrinsic_reward, state_features, new_state_features = self.local_agent.icm_act(state, new_state, action_one_hot, training=True)
                    total_reward = reward + intrinsic_reward
                    episode_reward += total_reward
                    Worker.global_steps += 1

                    mem.store(new_state,
                              total_reward,
                              done,
                              value[0, 0],
                              action_one_hot,
                              policy[0, action_index],
                              state_features, # (1, 288)
                              new_state_features)
                    episode_steps += 1

                    if done:
                        # calculate loss
                        agent_loss, forward_loss, inverse_loss = self.local_agent.compute_loss(mem, episode_reward, entropy_term)
                        self.update(tape, agent_loss, forward_loss, inverse_loss)
                        break

            # clear the experience
            mem.clear()
            # tape.reset()
            rewards.append(episode_reward)

            record(timestamp,
                   episode_reward,
                   self.worker_idx,
                   Worker.running_reward,
                   self.result_queue,
                   agent_loss,
                   episode_steps,
                   Worker.global_steps)

            # use a lock to save local model and to print to prevent data races.
            if episode_reward > Worker.best_score:
                print("Found better score: old = {}, new = {}".format(Worker.best_score, episode_reward))
                Worker.best_score = episode_reward
                # with Worker.save_lock:
                #     print('\nSaving best model to: {}, episode score: {}\n'.format(self.model_path, episode_reward))
                #     keras.models.save_model(self.global_agent, self.model_path)
                #     Worker.best_score = episode_reward

        self.result_queue.put(None)
        self.env.close()
