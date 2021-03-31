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
import tensorflow as tf
from tensorflow import keras
import threading
import time

from models.distributed_tf.gru import CnnGru
from models.common.util import ActionSpace, Memory
from models.common.util import record, instantiate_environment
from models.common.constants import *

matplotlib.use('agg')
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class DistributedMasterAgent():
    """DistributedMasterAgent: Generic distributed tensorflow based training
    """

    def __init__(self, env_path, train, evaluate, lr, timesteps, batch_size, gamma, save_dir, plot, eval_seeds=[]):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.plot = plot
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
        
        self.env = instantiate_environment(env_path, train, evaluate, eval_seeds)
        self.input_shape = self.env.observation_space[0].shape  # (84, 84, 3)

        # model parameters
        self.lr = lr
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.gamma = gamma
        self.model_path = os.path.join(self.save_dir, 'model_a3c_distributed')

        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        with self.mirrored_strategy.scope():
            # self.global_model = CNN(self.action_size, self.input_shape)
            self.global_model = CnnGru(self.action_size, self.input_shape)
            vec = np.random.random(self.input_shape)  # (84, 84, 3)
            vec = np.expand_dims(vec, axis=0)  # (1, 84, 84, 3)
            self.global_model(tf.convert_to_tensor(vec, dtype=tf.float32))

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './logs/' + self.current_time + '/distributed_tf/master'
        self.master_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # self.opt = tf.compat.v1.train.AdamOptimizer(lr, epsilon=1e-05, use_locking=True)
        self.opt = keras.optimizers.Adam(learning_rate=self.lr)

        
    def build_graph(self):
        """ build the model architecture """
        x = keras.Input(shape=(84, 84, 3))
        model = keras.Model(inputs=[x], outputs=self.global_model.call(x))
        keras.utils.plot_model(model, to_file=os.path.join(
            self.save_dir, 'model_a3c_architecture.png'), dpi=96, show_shapes=True, show_layer_names=True, expand_nested=False)
        return model

    def log_master_metrics(self, avg_reward, loss, step):
        with self.master_summary_writer.as_default():
            tf.summary.scalar('moving_average_reward', avg_reward, step=step)
            tf.summary.scalar('loss', loss, step=step)
            self.master_summary_writer.flush()
    
    def distributed_train(self):
        """ instantiate multiple workers and train the global model """
        start_time = time.time()
        res_queue = Queue()
        worker_params = {
            'env_path': self.env_path,
            'timesteps': self.timesteps,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'lr': self.lr,
            'optimizer': self.opt,
            'global_model': self.global_model,
            'master_summary_writer': self.master_summary_writer,
            'log_timestamp': self.current_time,
            'action_size': self.action_size,
            'action_lookup': self._action_lookup
        }
        
        worker = Worker(res_queue, worker_params, save_dir=self.save_dir)
        worker.start()

        # record episode reward to plot
        moving_average_rewards = []
        losses = []
        if self.plot:
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

        worker.join()
        end_time = time.time()

        print("\nTraining complete. Time taken = {} secs".format(
            end_time - start_time))

        if self.plot:
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
        model = keras.models.load_model(self.model_path, compile=False)
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
                action_index = np.random.choice(self.action_size, p=np.squeeze(policy))
                action = self._action_lookup[action_index]

                for i in range(4): # frame skipping
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
    episode_count = 0
    running_reward = 0
    best_score = 0
    global_steps = 0
    save_lock = threading.Lock()

    def __init__(self, result_queue, params, save_dir):
        super(Worker, self).__init__()
        self.result_queue = result_queue
        self.save_dir = save_dir
        self.model_path = os.path.join(self.save_dir, 'model_a3c_distributed')

        self.env = ObstacleTowerEnv(params['env_path'], worker_id=1,
                                    retro=False, realtime_mode=False, greyscale=False, config=train_env_reset_config)

        self.action_size = params['action_size']
        self._action_lookup = params['action_lookup']
        self.input_shape = self.env.observation_space[0].shape  # (84, 84, 3)
        self._last_health = 99999.
        self._last_keys = 0

        self.global_model = params['global_model']
        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        with self.mirrored_strategy.scope():
            # self.local_model = CNN(self.action_size, self.input_shape)
            self.local_model = CnnGru(self.action_size, self.input_shape)
    
        self.current_time = params['log_timestamp']
        train_log_dir = './logs/' + self.current_time + '/worker_1'
        self.worker_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        self.timesteps = params['timesteps']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.lr = params['lr']
        self.opt = params['optimizer']
        self.eps = np.finfo(np.float32).eps.item()
        self.ep_loss = 0.0

    def get_updated_reward(self, reward, new_health, new_keys, done):
        new_health = float(new_health)
        if done:  # penalize when game is terminated
            self._last_health = 99999.
            self._last_keys = 0
            reward = -1
        else:
            # crossing a floor- between [1, 4]
            if reward >= 1:
                reward += (new_health / 10000)
            
            # found time orb / crossed a floor
            if new_health > self._last_health:
                reward += 0.1

            # found a key
            if new_keys > self._last_keys:
                reward += 0.1

        return reward
    
    def log_worker_metrics(self, episode_reward, loss, step):
        with self.worker_summary_writer.as_default():
            tf.summary.scalar('reward', episode_reward, step=step)
            tf.summary.scalar('loss', loss, step=step)
            self.worker_summary_writer.flush()


    def run(self):
        mem = Memory()
        rewards = []
        ep_count = 1
        timestep = 0
        entropy_term = 0
        ep_reward = 0.
        ep_steps = 0
        ep_loss = 0.

        done = False
        obs = self.env.reset()
        state, _, _, _ = obs

        while timestep <= self.timesteps:
            with tf.GradientTape() as tape:
                for i in range(self.batch_size):
                    # collect experience
                    # get action as per policy
                    state = tf.convert_to_tensor(state)
                    state = tf.expand_dims(state, axis=0)
                    action_probs, critic_value = self.local_model(state, training=True)

                    entropy = -np.sum(action_probs * np.log(action_probs))
                    entropy_term += entropy

                    # choose most probable action
                    action_index = np.random.choice(
                        self.action_size, p=np.squeeze(action_probs))
                    action = self._action_lookup[action_index]

                    # perform action in game env
                    for i in range(4): # frame skipping
                        obs, reward, done, _ = self.env.step(action)
                        state, new_keys, new_health, cur_floor = obs
                        
                        reward = self.get_updated_reward(reward, new_health, new_keys, done)
                        self._last_health = new_health
                        self._last_keys = new_keys
                        
                        ep_reward += reward
                        ep_steps += 1
                        timestep += 1

                    # store experience
                    mem.store(action_prob=action_probs[0, action_index],
                              value=critic_value[0, 0],
                              reward=reward)

                    if done:
                        break
                
                # backpropagation
                total_loss = self.local_model.compute_loss(mem, state, done, self.gamma, self.eps, entropy_term)
                ep_loss += total_loss
                Worker.global_steps += ep_steps

            grads = tape.gradient(total_loss, self.local_model.trainable_variables)  # calculate local gradients
            # self.opt.apply_gradients(zip(grads, self.global_model.trainable_variables))  # send local gradients to global model
            # self.local_model.set_weights(self.global_model.get_weights())  # update local model with new weights
            mem.clear()
            
            if done:
                rewards.append(ep_reward)
                Worker.running_reward = sum(rewards[-10:]) / 10

                self.log_worker_metrics(ep_reward, ep_loss, ep_count)
                print("Episode: {} | Average Reward: {:.3f} | Episode Reward: {:.3f} | Loss: {:.3f} | Steps: {} | Total Steps: {} | Worker: {}".format(
                    Worker.episode_count, Worker.running_reward, ep_reward, ep_loss, ep_steps, Worker.global_steps, 1))
                self.result_queue.put((Worker.running_reward, total_loss))
                Worker.episode_count += 1
                ep_count += 1

                obs = self.env.reset()
                state, _, _, _ = obs
            
                # use a lock to save local model and to print to prevent data races.
                if ep_reward > Worker.best_score:
                    with Worker.save_lock:
                        print('\nSaving best model to: {}, episode score: {}\n'.format(self.model_path, ep_reward))
                        keras.models.save_model(self.global_model, self.model_path)
                        Worker.best_score = ep_reward
                
                entropy_term = 0
                ep_reward = 0.
                ep_steps = 0
                ep_loss = 0.
        
        self.result_queue.put(None)
        self.env.close()
