import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from queue import Queue
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
import threading

from models.util import record


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
        self.conv1 = layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), padding='same', activation='relu', data_format='channels_last', input_shape=self.ip_shape)    # (20, 20, 16)
        self.conv2 = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu', data_format='channels_last')                               # (9, 9, 32)
        self.flatten = layers.Flatten()                                                                                                                                         # (9 * 9 * 32)
        self.dense1 = layers.Dense(units=256, activation='relu')                                                                                                                # (256)

        # policy output layer (Actor)
        self.policy_logits = layers.Dense(self.action_size, name='policy_logits')                                                                                               # (54)

        # value output layer (Critic)
        self.values = layers.Dense(units=1, name='value')                                                                                                                       # (1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        
        logits = self.policy_logits(x)
        values = self.values(x)

        return logits, values


class MasterAgent():
    def __init__(self, env, lr, max_eps, update_freq, gamma, num_workers, save_dir):
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

        # TODO: check the syntax for our game with retro=False
        self.state_size = env.observation_space.shape[0]  # 84
        self.action_size = env.action_space.n  # 54
        self.input_shape = env.observation_space.shape  # (84, 84, 3)

        # TODO: replace optimizer with tf.keras.optimizers
        self.opt = tf.compat.v1.train.AdamOptimizer(lr, use_locking=True)

        # global network
        self.global_model = ActorCriticModel(self.state_size, self.action_size, self.input_shape)

        # TODO: check size
        # self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
        vec = np.random.random(self.input_shape) # (84, 84, 3)
        vec = np.expand_dims(vec, axis=0) # (1, 84, 84, 3)
        print("Input shape of image in global model: ", vec.shape)
        # vec = np.moveaxis(vec, 2, 0) # convert from channls last to channels first -> (3, 84, 84)
        self.global_model(tf.convert_to_tensor(vec, dtype=tf.float32))
        print(self.global_model.summary())
        # tf.keras.utils.plot_model(self.global_model, to_file=save_dir + 'model_architecture.png', show_shapes=True)

    def train(self):
        res_queue = Queue()

        workers = [Worker(self.state_size, self.action_size, self.global_model, self.opt, res_queue, i,
                          self.env, self.max_eps, self.update_freq, self.gamma, self.input_shape, save_dir=self.save_dir) for i in range(self.num_workers)]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        [w.join() for w in workers]

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average episode reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(self.save_dir, '{} Moving Average.png'.format(self.game_name)))
        # plt.show()

    def play(self):
        state = self.env.reset() # (84, 84, 3)
        state = np.expand_dims(state, axis=0) # (1, 84, 84, 3)
        model = self.global_model
        model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name))
        print('Loading model from: {}'.format(model_path))
        model.load_weights(model_path)
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                env.render(mode='rgb_array')
                policy, value = model(tf.convert_to_tensor(state, dtype=tf.float32))
                policy = tf.nn.softmax(policy)
                action = np.argmax(policy)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()


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
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self, state_size, action_size, global_model, opt, result_queue, idx, env, max_eps, update_freq, gamma, input_shape, game_name='OTC-v4.1', save_dir='/tmp'):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.input_shape = input_shape
        self.local_model = ActorCriticModel(self.state_size, self.action_size, self.input_shape)
        self.worker_idx = idx
        self.game_name = game_name

        # self.env = gym.make(self.game_name).unwrapped
        self.env = env
        self.save_dir = save_dir
        self.ep_loss = 0.0
        self.max_eps = max_eps
        self.update_freq = update_freq
        self.gamma = gamma

    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < self.max_eps:
            current_state = self.env.reset() # (84, 84, 3)
            # print("Current state: {}".format(current_state))
            # current_state = np.moveaxis(current_state, 2, 0) # convert from channls last to channels first -> (3, 84, 84)
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False
            while not done:
                # perform the forward pass
                logits, _ = self.local_model(tf.convert_to_tensor(np.expand_dims(current_state, axis=0), dtype=tf.float32))
                probs = tf.nn.softmax(logits)

                action = np.random.choice(self.action_size, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)
                
                if done:
                    reward = -1
                ep_reward += reward
                mem.store(current_state, action, reward)

                if time_count == self.update_freq or done:
                    # Calculate gradient wrt to local model
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done, new_state, mem, self.gamma)
                    
                    self.ep_loss += total_loss

                    # Calculate local gradients
                    # grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    grads = tape.gradient(total_loss, self.local_model.trainable_variables)

                    # Push local gradients to global model
                    # self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))
                    self.opt.apply_gradients(zip(grads, self.global_model.trainable_variables))

                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())

                    mem.clear()
                    time_count = 0

                    if done:  # done and print information
                        Worker.global_moving_average_reward = \
                            record(Worker.global_episode, ep_reward, self.worker_idx,
                                   Worker.global_moving_average_reward, self.result_queue, self.ep_loss, ep_steps)

                        # We must use a lock to save our model and to print to prevent data races.
                        if ep_reward > Worker.best_score:
                            with Worker.save_lock:
                                print("Saving best model to {}, episode score: {}".format(self.save_dir, ep_reward))
                                self.global_model.save_weights(os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name)))
                                Worker.best_score = ep_reward
                        Worker.global_episode += 1
                ep_steps += 1

                time_count += 1
                current_state = new_state
                total_step += 1
        self.result_queue.put(None)

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        if done:
            reward_sum = 0.  # terminal
        else:
            reward_sum = self.local_model(tf.convert_to_tensor(np.expand_dims(new_state, axis=0), dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        # TODO: try to normalize the discounted rewards

        logits, values = self.local_model(tf.convert_to_tensor(np.stack(memory.states), dtype=tf.float32))
        
        # Get our advantages
        q_value_estimate = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32)
        advantage = q_value_estimate - values
        
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        actions_one_hot = tf.one_hot(memory.actions, self.action_size, dtype=tf.float32)

        policy = tf.nn.softmax(logits)
        # entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)
        entropy = tf.reduce_sum(policy * tf.math.log(policy + 1e-20), axis=1)

        # policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions, logits=logits)
        policy_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=actions_one_hot, logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy

        # total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))

        return total_loss
