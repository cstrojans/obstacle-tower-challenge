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
import tensorflow_probability as tfp
from tensorflow.python import keras
from tensorflow.python.keras import layers
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
    def __init__(self, env_path=None, train=False, evaluate=False, eval_seeds=[], lr=0.001, max_eps=10, update_freq=20, gamma=0.99, num_workers=0, save_dir='./model_files/'):
        self.game_name = 'OTC-v4.1'
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.env_path = env_path
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

        # TODO: check the syntax for our game with retro=False
        # self.action_size = self.env.action_space.n  # 54
        self.action_size = 10
        # self._action_lookup = {
        #     0: np.asarray([0, 0, 0, 0]),  # nop
        #     1: np.asarray([1, 0, 0, 0]),  # forward
        #     2: np.asarray([2, 0, 0, 0]),  # backward
        #     3: np.asarray([0, 1, 0, 0]),  # cam left
        #     4: np.asarray([0, 2, 0, 0]),  # cam right
        #     5: np.asarray([1, 0, 1, 0]),   # jump forward
        #     6: np.asarray([1, 1, 0, 0]),  # forward + cam left
        #     7: np.asarray([1, 2, 0, 0])  # forward + cam right
        # }

        self._action_lookup = {
            0: np.asarray([1, 0, 0, 0]),  # forward
            1: np.asarray([2, 0, 0, 0]),  # backward
            2: np.asarray([0, 0, 0, 1]),  # left
            3: np.asarray([0, 0, 0, 2]),  # right
            4: np.asarray([1, 0, 1, 0]),   # jump forward
            5: np.asarray([2, 0, 1, 0]),   # jump backward
            6: np.asarray([1, 1, 0, 0]),  # forward + cam left
            7: np.asarray([1, 2, 0, 0]),  # forward + cam right
            8: np.asarray([2, 1, 0, 0]),  # backward + cam left
            9: np.asarray([2, 2, 0, 0])  # backward + cam right
        }

        self.input_shape = self.env.observation_space[0].shape  # (84, 84, 3)

        # model parameters
        self.lr = lr
        self.max_eps = max_eps
        self.update_freq = update_freq
        self.gamma = gamma
        if num_workers == 0:
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = num_workers
        self.model_path = os.path.join(self.save_dir, 'model_a3c')
        self.opt = tf.compat.v1.train.AdamOptimizer(
            lr, epsilon=1e-05, use_locking=True)
        # self.opt = tf.keras.optimizers.RMSprop(learning_rate=lr, epsilon=1e-05, use_locking=True)

        # self.global_model = CNN(self.action_size, self.input_shape)
        self.global_model = CnnGru(self.action_size, self.input_shape)

        vec = np.random.random(self.input_shape)  # (84, 84, 3)
        vec = np.expand_dims(vec, axis=0)  # (1, 84, 84, 3)
        self.global_model(tf.convert_to_tensor(vec, dtype=tf.float32))

        # store model architecture in image
        tf.keras.utils.plot_model(self.build_graph(), to_file=os.path.join(
            self.save_dir, 'model_a3c_architecture.png'), dpi=96, show_shapes=True, show_layer_names=True, expand_nested=False)

    def build_graph(self):
        x = tf.keras.Input(shape=(84, 84, 3))
        return tf.keras.Model(inputs=[x], outputs=self.global_model.call(x))

    def train(self):
        start_time = time.time()

        res_queue = Queue()

        workers = [Worker(self.action_size, self.global_model, self.opt, res_queue, worker_id, self.env_path, self.max_eps,
                          self.update_freq, self.gamma, self.input_shape, save_dir=self.save_dir) for worker_id in range(1, self.num_workers+1)]

        for i, worker in enumerate(workers, start=1):
            print("Starting worker {}".format(i))
            worker.start()

        # record episode reward to plot
        moving_average_rewards = []
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        for w in workers:
            w.join()

        end_time = time.time()
        print("\nTraining complete. Time taken = {} secs".format(
            end_time - start_time))

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average episode reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(self.save_dir,
                                 'model_a3c_moving_average.png'))

        # save the trained model to a file
        print('Saving model to: {}'.format(self.model_path))
        tf.keras.models.save_model(self.global_model, self.model_path)
        self.env.close()

    def play_single_episode(self):
        action_space = ActionSpace()
        print('Loading model from: {}'.format(self.model_path))
        model = tf.keras.models.load_model(self.model_path, compile=False)
        print("Playing single episode...")
        done = False
        step_counter = 0
        reward_sum = 0
        obs = self.env.reset()
        state, _, _, _ = obs

        try:
            while not done:
                policy, value = model(tf.convert_to_tensor(
                    np.expand_dims(state, axis=0), dtype=tf.float32))
                policy = tf.nn.softmax(policy)
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
        # run episodes until evaluation is complete
        while not self.env.evaluation_complete:
            episode_reward = self.play_single_episode()

        pprint(self.env.results)
        self.env.close()


class Worker(threading.Thread):
    global_episode = 0
    global_moving_average_reward = 0
    best_score = 0
    global_steps = 0
    save_lock = threading.Lock()

    def __init__(self, action_size, global_model, opt, result_queue, idx, env_path, max_eps,
                 update_freq, gamma, input_shape, game_name='OTC-v4.1', save_dir='./model_files/'):
        super(Worker, self).__init__()
        self.result_queue = result_queue
        self.worker_idx = idx
        self.game_name = game_name

        self.env = ObstacleTowerEnv(env_path, worker_id=self.worker_idx,
                                    retro=False, realtime_mode=False, greyscale=False, config=train_env_reset_config)

        # self.action_size = self.env.action_space.n  # 54
        self.action_size = action_size
        # self._action_lookup = {
        #     0: np.asarray([0, 0, 0, 0]),  # nop
        #     1: np.asarray([1, 0, 0, 0]),  # forward
        #     2: np.asarray([2, 0, 0, 0]),  # backward
        #     3: np.asarray([0, 1, 0, 0]),  # cam left
        #     4: np.asarray([0, 2, 0, 0]),  # cam right
        #     5: np.asarray([1, 0, 1, 0]),   # jump forward
        #     6: np.asarray([1, 1, 0, 0]),  # forward + cam left
        #     7: np.asarray([1, 2, 0, 0])  # forward + cam right
        # }
        self._action_lookup = {
            0: np.asarray([1, 0, 0, 0]),  # forward
            1: np.asarray([2, 0, 0, 0]),  # backward
            2: np.asarray([0, 0, 0, 1]),  # left
            3: np.asarray([0, 0, 0, 2]),  # right
            4: np.asarray([1, 0, 1, 0]),   # jump forward
            5: np.asarray([2, 0, 1, 0]),   # jump backward
            6: np.asarray([1, 1, 0, 0]),  # forward + cam left
            7: np.asarray([1, 2, 0, 0]),  # forward + cam right
            8: np.asarray([2, 1, 0, 0]),  # backward + cam left
            9: np.asarray([2, 2, 0, 0])  # backward + cam right
        }

        self.input_shape = self.env.observation_space[0].shape  # (84, 84, 3)
        self._last_health = 99999.
        self._last_keys = 0

        # self.local_model = CNN(self.action_size, self.input_shape)
        self.local_model = CnnGru(self.action_size, self.input_shape)
        self.global_model = global_model
        self.opt = opt
        self.input_shape = input_shape

        self.save_dir = save_dir
        self.model_path = os.path.join(self.save_dir, 'model_a3c')
        self.ep_loss = 0.0
        self.max_eps = max_eps
        self.update_freq = update_freq
        self.gamma = gamma

    def get_updated_reward(self, reward, new_health, new_keys):
        if reward >= 1:  # give more reward to prioritize crossing a level
            reward += (new_health / 10000)
        elif new_health > self._last_health:  # found time orb
            reward = 0.1

        if new_keys > self._last_keys:
            reward = 1

        return reward

    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < self.max_eps:
            obs = self.env.reset()
            current_state, _, _, _ = obs
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0
            time_count = 0
            done = False

            while not done:
                # get action as per the policy
                logits, _ = self.local_model(tf.convert_to_tensor(
                    np.expand_dims(current_state, axis=0), dtype=tf.float32))
                probs = tf.nn.softmax(logits)

                # perform action in env and get next state
                action_index = np.random.choice(
                    self.action_size, p=probs.numpy()[0])
                action = self._action_lookup[action_index]
                obs, reward, done, _ = self.env.step(action)
                new_state, new_keys, new_health, cur_floor = obs
                new_health = float(new_health)
                # get total number of keys pressed from the MultiDiscrete space
                new_keys = int(new_keys.sum())

                reward = self.get_updated_reward(reward, new_health, new_keys)
                self._last_health = new_health
                self._last_keys = new_keys

                if done:
                    self._last_health = 99999.
                    reward = -1

                ep_reward += reward
                mem.store(current_state, action, reward)

                if time_count == self.update_freq or done:
                    # calculate gradient wrt to local model
                    with tf.GradientTape() as tape:
                        total_loss = self.local_model.compute_loss(
                            done, new_state, mem, self.gamma)

                    self.ep_loss += total_loss

                    # calculate local gradients
                    grads = tape.gradient(
                        total_loss, self.local_model.trainable_variables)

                    # send local gradients to global model
                    self.opt.apply_gradients(
                        zip(grads, self.global_model.trainable_variables))

                    # update local model with new weights
                    self.local_model.set_weights(
                        self.global_model.get_weights())

                    mem.clear()
                    time_count = 0

                    if done:
                        Worker.global_moving_average_reward = \
                            record(Worker.global_episode, ep_reward, self.worker_idx,
                                   Worker.global_moving_average_reward, self.result_queue, self.ep_loss, ep_steps, Worker.global_steps)

                        # use a lock to save local model and to print to prevent data races.
                        if ep_reward > Worker.best_score:
                            with Worker.save_lock:
                                print('\nSaving best model to: {}, episode score: {}\n'.format(
                                    self.model_path, ep_reward))
                                tf.keras.models.save_model(
                                    self.global_model, self.model_path)

                                Worker.best_score = ep_reward
                        Worker.global_episode += 1

                ep_steps += 1
                time_count += 1
                current_state = new_state
                total_step += 1
                Worker.global_steps += 1

        self.result_queue.put(None)
        self.env.close()
