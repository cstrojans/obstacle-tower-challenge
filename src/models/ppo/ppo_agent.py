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
import tensorboard
from tensorflow import keras
import threading
import time

from models.ppo.cnn import CNN
from models.ppo.gru import CnnGru
from models.common.util import ActionSpace, Memory_PPO
from models.common.util import record, instantiate_environment
from models.common.constants import *

matplotlib.use('agg')
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class MasterAgent():
    """MasterAgent A3C: Asynchronous Advantage Actor Critic Model is a model-free policy gradient algorithm.
    """

    def __init__(self, env_path, train, evaluate, lr, timesteps, batch_size, gamma, num_workers, save_dir, eval_seeds=[]):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.env_path = env_path
        self.action_size = 7
        self._action_lookup = {  # action space from fourth place winner
            0: np.asarray([1, 0, 0, 0]),  # forward
            1: np.asarray([2, 0, 0, 0]),  # backward
            2: np.asarray([0, 1, 0, 0]),  # cam left
            3: np.asarray([0, 2, 0, 0]),  # cam right
            4: np.asarray([1, 0, 1, 0]),  # forward + jump
            5: np.asarray([1, 1, 0, 0]),  # forward + cam left
            6: np.asarray([1, 2, 0, 0]),  # forward + cam right
        }

        self.num_workers = multiprocessing.cpu_count() if num_workers == 0 else num_workers
        self.env = instantiate_environment(env_path, train, evaluate, eval_seeds)
        self.input_shape = self.env.observation_space[0].shape  # (84, 84, 3)

        # model parameters
        self.lr = lr
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.gamma = gamma
        self.model_path = os.path.join(self.save_dir, 'model_ppo')
        
        # self.global_model = CNN(self.action_size, self.input_shape)
        self.global_model = CnnGru(self.action_size, self.input_shape)

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './logs/' + self.current_time + '/ppo/master'
        self.master_summary_writer = tf.summary.create_file_writer(train_log_dir)

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr, decay_steps=1000, decay_rate=0.9)
        self.opt = keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-05)

        vec = np.random.random(self.input_shape)  # (84, 84, 3)
        vec = np.expand_dims(vec, axis=0)  # (1, 84, 84, 3)
        self.global_model(tf.convert_to_tensor(vec, dtype=tf.float32), training=True)

    def build_graph(self):
        """ build the model architecture """
        x = keras.Input(shape=(84, 84, 3))
        model = keras.Model(inputs=[x], outputs=self.global_model.call(x))
        keras.utils.plot_model(model, to_file=os.path.join(
            self.save_dir, 'model_ppo_architecture.png'), dpi=96, show_shapes=True, show_layer_names=True, expand_nested=False)
        return model

    def log_master_metrics(self, avg_reward, loss, step):
        with self.master_summary_writer.as_default():
            tf.summary.scalar('moving_average_reward', avg_reward, step=step)
            tf.summary.scalar('loss', loss, step=step)
            self.master_summary_writer.flush()

    def train(self):
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
        workers = [Worker(res_queue,
                          worker_id,
                          self.save_dir,
                          worker_params) for worker_id in range(1, self.num_workers+1)]

        for i, worker in enumerate(workers, start=1):
            print("Starting worker {}".format(i))
            worker.start()

        # record episode reward to plot
        moving_average_rewards = []
        losses = []
        i = 0
        while True:
            result = res_queue.get()
            i += 1
            if result:
                reward, loss = result
            else:
                break
            if reward is None or not loss:
                break
            else:
                self.log_master_metrics(reward, loss, i)
                moving_average_rewards.append(reward)
                losses.append(loss)
        
        for w in workers:
            w.join()

        end_time = time.time()
        print("\nTraining complete. Time taken = {} secs".format(
            end_time - start_time))

        # save the trained model to a file
        # print('Saving global model to: {}'.format(self.model_path))
        # keras.models.save_model(self.global_model, self.model_path)

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

    def __init__(self, result_queue, idx, save_dir, params):
        super(Worker, self).__init__()
        self.result_queue = result_queue
        self.worker_idx = idx
        self.save_dir = save_dir
        self.model_path = os.path.join(self.save_dir, 'model_ppo')

        self.env = ObstacleTowerEnv(params['env_path'], worker_id=self.worker_idx,
                                    retro=False, realtime_mode=False, greyscale=False, config=train_env_reset_config)

        self.action_size = params['action_size']
        self._action_lookup = params['action_lookup']
        self.input_shape = self.env.observation_space[0].shape  # (84, 84, 3)
        self._last_health = 99999.
        self._last_keys = 0

        self.global_model = params['global_model']
        # self.local_model = CNN(self.action_size, self.input_shape)
        self.local_model = CnnGru(self.action_size, self.input_shape)
        
        self.current_time = params['log_timestamp']
        train_log_dir = './logs/' + self.current_time + '/worker_' + str(self.worker_idx)
        self.worker_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        self.timesteps = params['timesteps']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.lr = params['lr']
        self.opt = params['optimizer']
        self.eps = np.finfo(np.float32).eps.item()

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
        horizon = 512
        batch_size = horizon // 16
        n_epochs = 24
        mem = Memory_PPO(batch_size=batch_size)
        n_steps = 0
        rewards = []
        ep_count = 1
        timestep = 0
        entropy_term = 0
        ep_reward = 0.
        ep_steps = 0
        ep_loss = 0.

        done = False
        obs = self.env.reset()
        state, new_keys, new_health, cur_floor = obs

        while timestep <= self.timesteps:
            for i in range(self.batch_size):
                # collect experience
                # get action as per policy
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, axis=0)
                action_probs, critic_value = self.local_model(state, training=True)

                # entropy = -np.sum(action_probs * np.log(action_probs))
                # entropy_term += entropy

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
                n_steps += 1
                # store experience
                mem.store(state=state,
                    action=action,
                    action_prob=action_probs,
                    value=critic_value[0, 0],
                    reward=reward,
                    is_terminal=done)

                if done or n_steps % horizon == 0:
                    # backpropagation
                    total_loss = self.run_training(mem, self.gamma, self.eps, self.opt, n_epochs, batch_size)
                    ep_loss += total_loss
                    Worker.global_steps += ep_steps        
                    mem.clear()

                    if done:
                        break
            
            if done:
                rewards.append(ep_reward)
                Worker.running_reward = sum(rewards[-10:]) / 10

                self.log_worker_metrics(ep_reward, ep_loss, ep_count)
                print("Episode: {} | Average Reward: {:.3f} | Episode Reward: {:.3f} | Loss: {:.3f} | Steps: {} | Total Steps: {} | Worker: {}".format(
                    Worker.episode_count, Worker.running_reward, ep_reward, ep_loss, ep_steps, Worker.global_steps, self.worker_idx))
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
    
    def run_training(self, memory, gamma, eps, loss_fn, n_epochs, batch_size):
        n_epochs = 3
        clipping_val = 0.2
        critic_discount = 0.5
        entropy_beta = 0.01
        gamma = 0.9975
        gae_lambda = 0.95
        loss_sum = 0
        for _ in range(n_epochs):
            state_arr, action_arr, old_prob_arr, \
            vals_arr, reward_arr, dones_arr, batches = \
                memory.generate_batches()
            
            values = vals_arr
            advantage = np.zeros((len(reward_arr), len(old_prob_arr[0])), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount*(reward_arr[k] + gamma*values[k+1]*\
                                (1-int(dones_arr[k])) - values[k])
                    discount *= gamma*gae_lambda
                for i in range(len(old_prob_arr[0])):
                    advantage[t][i] = a_t
            
            for batch in batches:
                with tf.GradientTape() as tape:
                    old_probs = tf.math.log(tf.convert_to_tensor(np.array(old_prob_arr[batch])))
                    actor_action_probs = []
                    critic_values = []
                    for st in state_arr[batch]:
                        probs, vals = self.local_model(np.expand_dims(np.array(st), axis=0))
                        actor_action_probs.append(probs)
                        critic_values.append(vals)
                    
                    
                    actor_action_probs = tf.stack(actor_action_probs)
                    critic_values = tf.stack(critic_values)
                    # print(actor_action_probs.shape, old_probs.shape)
                    log_probs = tf.math.log(actor_action_probs)
                    prob_ratio = tf.math.exp(log_probs - old_probs)
                    # prob_ratio = tf.math.exp(log_probs) / tf.math.exp(old_probs)

                    # print(advantage[batch].shape, prob_ratio.shape)
                    weighted_probs = advantage[batch] * prob_ratio
                    # print(weighted_probs.shape, advantage[batch][0], prob_ratio, weighted_probs[0])
                    weighted_clipped_probs = tf.clip_by_value(prob_ratio, 1 - clipping_val, 1 + clipping_val) * advantage[batch]
                    actor_loss = - tf.reduce_mean(tf.math.minimum(weighted_probs, weighted_clipped_probs))

                    returns =  advantage[batch] + values[batch]
                    critic_loss = tf.reduce_mean((returns - critic_values) ** 2)

                    entropy = -tf.reduce_mean(actor_action_probs * tf.math.log(actor_action_probs))
                    total_loss = actor_loss + critic_discount * critic_loss \
                         - entropy_beta * entropy
                    # print("Actor Loss: {}, Critic_Loss: {}, Entropy: {}".format(actor_loss, critic_loss, entropy))
                    grads, _ = tf.clip_by_global_norm(tape.gradient(total_loss, self.local_model.trainable_variables), 1.0)  # calculate local gradients
                    # grads = tape.gradient(total_loss, self.local_model.trainable_variables)  # calculate local gradients
                    self.opt.apply_gradients(zip(grads, self.global_model.trainable_variables))  # send local gradients to global model
                    self.local_model.set_weights(self.global_model.get_weights())  # update local model with new weights

                    loss_sum += total_loss
        return loss_sum