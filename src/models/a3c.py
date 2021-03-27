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

from models.architecture.cnn import CNN
from models.architecture.gru import CnnGru
from models.util import ActionSpace, Memory
from models.util import record
from definitions import *

matplotlib.use('agg')
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class MasterAgent():
    """MasterAgent A3C: Asynchronous Advantage Actor Critic Model is a model-free policy gradient algorithm.
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
        # self.global_model = CNN(self.action_size, self.input_shape)
        self.global_model = CnnGru(self.action_size, self.input_shape)

        # self.opt = tf.compat.v1.train.AdamOptimizer(lr, epsilon=1e-05, use_locking=True)
        self.opt = keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = keras.losses.Huber()

        vec = np.random.random(self.input_shape)  # (84, 84, 3)
        vec = np.expand_dims(vec, axis=0)  # (1, 84, 84, 3)
        self.global_model(tf.convert_to_tensor(vec, dtype=tf.float32))

    def build_graph(self):
        """ build the model architecture """
        x = keras.Input(shape=(84, 84, 3))
        model = keras.Model(inputs=[x], outputs=self.global_model.call(x))
        keras.utils.plot_model(model, to_file=os.path.join(
            self.save_dir, 'model_a3c_architecture.png'), dpi=96, show_shapes=True, show_layer_names=True, expand_nested=False)
        return model

    def train(self):
        """ instantiate multiple workers and train the global model """
        start_time = time.time()
        res_queue = Queue()
        workers = [Worker(self.action_size,
                          self._action_lookup,
                          self.global_model,
                          self.opt,
                          self.loss_fn,
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
            if not reward or not loss:
                break
            else:
                moving_average_rewards.append(reward)
                losses.append(loss)
            
        for w in workers:
            w.join()

        end_time = time.time()
        print("\nTraining complete. Time taken = {} secs".format(
            end_time - start_time))

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        fig.suptitle('A3C Model')

        ax1.plot(range(1, len(moving_average_rewards) + 1), moving_average_rewards)
        ax1.set_title('Reward vs Timesteps')
        ax1.set(xlabel='Episodes', ylabel='Reward')

        ax2.plot(range(1, len(losses) + 1), losses)
        ax2.set_title('Loss vs Timesteps')
        ax2.set(xlabel='Episodes', ylabel='Loss')
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, 'model_a3c_moving_average.png'))

        # save the trained model to a file
        print('Saving global model to: {}'.format(self.model_path))
        keras.models.save_model(self.global_model, self.model_path)
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
        obs = self.env.reset()
        state, _, _, _ = obs

        try:
            while not done:
                policy, _ = model(tf.convert_to_tensor(
                    np.expand_dims(state, axis=0), dtype=tf.float32))
                action_index = np.argmax(policy)
                action = self._action_lookup[action_index]

                obs, reward, done, _ = self.env.step(action)
                state, _, _, _ = obs
                reward_sum += reward

                print("{}. Reward: {}, action: {}".format(step_counter,
                                                          reward_sum, action_space.get_action_meaning(action)))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            if not self.evaluate:
                self.env.close()
            return reward_sum

    def evaluate(self):
        """ run episodes until evaluation is complete """
        while not self.env.evaluation_complete:
            episode_reward = self.play_single_episode()

        pprint(self.env.results)
        self.env.close()


class Worker(threading.Thread):
    episode_count = 0
    running_reward = 0
    best_score = 0
    global_steps = 0
    save_lock = threading.Lock()

    def __init__(self, action_size, action_lookup, global_model, opt_fn, loss_fn, max_eps,
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

        self.global_model = global_model
        # self.local_model = CNN(self.action_size, self.input_shape)
        self.local_model = CnnGru(self.action_size, self.input_shape)
        self.opt = opt_fn
        self.loss_fn = loss_fn
        self.ep_loss = 0.0
        self.max_eps = max_eps
        self.update_freq = update_freq
        self.gamma = gamma
        # smallest number such that 1.0 + eps != 1.0
        self.eps = np.finfo(np.float32).eps.item()
        self.model_path = os.path.join(self.save_dir, 'model_a3c')

    def get_updated_reward(self, reward, new_health, new_keys, done):
        new_health = float(new_health)
        if done:  # penalize when game is terminated
            self._last_health = 99999.
            self._last_keys = 0
            reward = -1
        else:
            if reward >= 1:  # prioritize crossing a floor/level - between [1, 4]
                reward += (new_health / 10000)
            elif new_health > self._last_health:  # found time orb
                reward = 0.1

            if new_keys > self._last_keys:  # found a key
                reward = 1

        return reward

    def run(self):
        mem = Memory()
        while Worker.episode_count < self.max_eps:
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0.
            done = False

            obs = self.env.reset()
            state, _, _, _ = obs

            with tf.GradientTape() as tape:
                for timestep in range(1, self.update_freq):
                    state = tf.convert_to_tensor(state)
                    state = tf.expand_dims(state, axis=0)
                    action_probs, critic_value = self.local_model(state)

                    action_index = np.random.choice(
                        self.action_size, p=np.squeeze(action_probs))
                    action = self._action_lookup[action_index]
                    obs, reward, done, _ = self.env.step(action)
                    state, new_keys, new_health, cur_floor = obs
                    
                    # reward = self.get_updated_reward(reward, new_health, new_keys, done)
                    # self._last_health = new_health
                    # self._last_keys = new_keys

                    mem.store(action_prob=action_probs[0, action_index],
                              value=critic_value[0, 0],
                              reward=reward)
                    ep_reward += reward
                    ep_steps += 1
                    Worker.global_steps += 1

                    if done:
                        break
                
                Worker.running_reward = 0.05 * ep_reward + (1 - 0.05) * Worker.running_reward

                # backpropagation
                total_loss = self.local_model.compute_loss(mem, state, done, self.gamma, self.eps, self.loss_fn)
                grads = tape.gradient(total_loss, self.local_model.trainable_variables)  # calculate local gradients
                self.opt.apply_gradients(zip(grads, self.global_model.trainable_variables))  # send local gradients to global model
                self.local_model.set_weights(self.global_model.get_weights())  # update local model with new weights
                
                self.ep_loss += total_loss
                mem.clear()
            
            Worker.episode_count += 1
            Worker.running_reward = record(Worker.episode_count,
                                        ep_reward,
                                        self.worker_idx,
                                        Worker.running_reward,
                                        self.result_queue,
                                        self.ep_loss,
                                        ep_steps,
                                        Worker.global_steps)

            # use a lock to save local model and to print to prevent data races.
            if ep_reward > Worker.best_score:
                with Worker.save_lock:
                    print('\nSaving best model to: {}, episode score: {}\n'.format(self.model_path, ep_reward))
                    keras.models.save_model(self.global_model, self.model_path)
                    Worker.best_score = ep_reward
                
        self.result_queue.put(None)
        self.env.close()
