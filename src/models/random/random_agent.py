from queue import Queue
import gym
import matplotlib
import matplotlib.pyplot as plt
from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import os
from prettyprinter import pprint
import time

from models.common.constants import *
from models.common.util import *


matplotlib.use('agg')


class RandomAgent:
    """Random Agent that will play the specified game
      Args:
        env_name: Name of the environment to be played
        max_eps: Maximum number of episodes to run agent for.
    """

    def __init__(self, env_path, train=False, evaluate=False, eval_seeds=[], max_eps=100, save_dir=None, plot=False):
        if train:
            self.env = ObstacleTowerEnv(
                env_path, worker_id=0, retro=False, realtime_mode=False, config=train_env_reset_config)
        else:
            if evaluate:
                self.env = ObstacleTowerEnv(
                    env_path, worker_id=0, retro=False, realtime_mode=False, config=eval_env_reset_config)
                self.env = ObstacleTowerEvaluation(self.env, eval_seeds)
            else:
                self.env = ObstacleTowerEnv(
                    env_path, worker_id=0, retro=False, realtime_mode=True, config=eval_env_reset_config)
        self.max_episodes = max_eps
        self.global_moving_average_reward = 0
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.plot = plot
        self.res_queue = Queue()

    def train(self):
        start_time = time.time()
        reward_avg = 0
        global_steps = 0
        moving_average_rewards = []
        for episode in range(self.max_episodes):
            done = False
            self.env.reset()
            reward_sum = 0.0
            steps = 0
            while not done:
                # Sample randomly from the action space and step
                _, reward, done, _ = self.env.step(
                    self.env.action_space.sample())
                steps += 1
                global_steps += 1
                reward_sum += reward

            if self.plot:
                # Record statistics
                moving_average_rewards.append(reward_sum)

            reward_avg += reward_sum
            self.global_moving_average_reward = record(
                episode, reward_sum, 0, self.global_moving_average_reward, self.res_queue, 0, steps, global_steps)
        end_time = time.time()
        print("\nTraining complete. Time taken = {} secs".format(end_time - start_time))
        final_avg = reward_avg / float(self.max_episodes)
        print("Average score across {} episodes: {}".format(
            self.max_episodes, final_avg))

        if self.plot:
            plt.plot(moving_average_rewards)
            plt.ylabel('Moving average episode reward')
            plt.xlabel('Step')
            plt.savefig(os.path.join(self.save_dir,
                                     'model_random_moving_average.png'))

        self.env.close()
        return final_avg

    def play_single_episode(self):
        action_space = ActionSpace()
        print("Playing single episode...")
        done = False
        step_counter = 0
        reward_sum = 0
        obs = self.env.reset()
        state, _, _, _ = obs

        try:
            while not done:
                action = self.env.action_space.sample()
                obs, reward, done, info = self.env.step(action)
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

