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
from tensorflow.python import keras
from tensorflow.python.keras import layers
import threading
import time

from models.util import *
from definitions import *

matplotlib.use('agg')
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size, ip_shape=(84, 84, 3)):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.ip_shape = ip_shape

        # CNN - spatial dependencies
        """
        self.conv1 = layers.Conv2D(filter=32, kernel_size=(8, 8), strides=(
            4, 4), padding='same', activation='relu', input_shape=self.input_shape)
        self.conv2 = layers.Conv2D(filter=64, kernel_size=(
            4, 4), strides=(2, 2), padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filter=64, kernel_size=(
            3, 3), strides=(1, 1), padding='same', activation='relu')
        # self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(units=512, activation='relu')

        # TODO: LSTM - temporal dependencies
        # self.lstm1 = layers.LSTM(units=256, return_sequences=False)
        """
        
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


class MasterAgent():
    def __init__(self, env_path=None, train=False, evaluate=False, eval_seeds=[], lr=0.001, max_eps=10, update_freq=20, gamma=0.99, num_workers=0, save_dir='./model_files/'):
        self.game_name = 'OTC-v4.1'
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.env = env
        self.lr = lr
        self.max_eps = max_eps
        self.update_freq = update_freq
        self.gamma = gamma
        self.num_workers = num_workers
        self.model_path = os.path.join(self.save_dir, 'model_a3c')

        # TODO: check the syntax for our game with retro=False
        self.state_size = env.observation_space.shape[0]  # 84
        self.action_size = env.action_space.n  # 54
        self.input_shape = env.observation_space.shape  # (84, 84, 3)

        # TODO: replace optimizer with tf.keras.optimizers
        # TODO: try RMSProp optimizer instead of Adam
        self.opt = tf.compat.v1.train.AdamOptimizer(lr, use_locking=True)

        # global network
        self.global_model = ActorCriticModel(
            self.state_size, self.action_size, self.input_shape)

        vec = np.random.random(self.input_shape)  # (84, 84, 3)
        vec = np.expand_dims(vec, axis=0)  # (1, 84, 84, 3)
        self.global_model(tf.convert_to_tensor(vec, dtype=tf.float32))

        tf.keras.utils.plot_model(self.build_graph(), to_file=os.path.join(
            self.save_dir, 'model_a3c_architecture.png'), dpi=96, show_shapes=True, show_layer_names=True, expand_nested=False)

    def build_graph(self):
        x = tf.keras.Input(shape=(84, 84, 3))
        return tf.keras.Model(inputs=[x], outputs=self.global_model.call(x))

    def train(self):
        start_time = time.time()
        
        res_queue = Queue()

        workers = [Worker(self.state_size, self.action_size, self.global_model, self.opt, res_queue, worker_id, self.env_path, self.max_eps,
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
        print("\nTraining complete. Time taken = {} secs".format(end_time - start_time))

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
        state = self.env.reset()

        try:
            while not done:
                policy, value = model(tf.convert_to_tensor(
                    np.expand_dims(state, axis=0), dtype=tf.float32))
                policy = tf.nn.softmax(policy)
                action = np.argmax(policy)
                state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(
                    step_counter, reward_sum, action_space.get_action_meaning(action)))
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


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Worker(threading.Thread):
    global_episode = 0
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self, state_size, action_size, global_model, opt, result_queue, idx, env_path, max_eps,
                 update_freq, gamma, input_shape, game_name='OTC-v4.1', save_dir='./model_files/'):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.input_shape = input_shape
        self.local_model = ActorCriticModel(
            self.state_size, self.action_size, self.input_shape)
        self.worker_idx = idx
        self.game_name = game_name
        self.env = ObstacleTowerEnv(env_path, worker_id=self.worker_idx, retro=True, realtime_mode=False, config=train_env_reset_config)
        self.save_dir = save_dir
        self.model_path = os.path.join(self.save_dir, 'model_a3c_local')
        self.ep_loss = 0.0
        self.max_eps = max_eps
        self.update_freq = update_freq
        self.gamma = gamma

    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < self.max_eps:
            current_state = self.env.reset()
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

                action = np.random.choice(self.action_size, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)

                if done:
                    reward = -1
                ep_reward += reward
                mem.store(current_state, action, reward)

                if time_count == self.update_freq or done:
                    # calculate gradient wrt to local model
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(
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
                                   Worker.global_moving_average_reward, self.result_queue, self.ep_loss, ep_steps)

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

        self.result_queue.put(None)
        self.env.close()

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        beta_regularizer = 0.01

        if done:  # game has terminated
            reward_sum = 0.
        else:
            reward_sum = self.local_model(tf.convert_to_tensor(
                np.expand_dims(new_state, axis=0), dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        # TODO: try to normalize the discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        # TODO: try to normalize the discounted rewards

        logits, values = self.local_model(tf.convert_to_tensor(
            np.stack(memory.states), dtype=tf.float32))

        # get our advantages
        q_value_estimate = tf.convert_to_tensor(
            np.array(discounted_rewards)[:, None], dtype=tf.float32)
        advantage = q_value_estimate - values

        # value loss
        value_loss = advantage ** 2

        # policy loss
        actions_one_hot = tf.one_hot(
            memory.actions, self.action_size, dtype=tf.float32)

        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=policy, logits=logits)
        # entropy = tf.reduce_sum(policy * tf.math.log(policy + 1e-20), axis=1)

        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=memory.actions, logits=logits)
        # policy_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        #     labels=actions_one_hot, logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= beta_regularizer * entropy

        # total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))

        return total_loss
