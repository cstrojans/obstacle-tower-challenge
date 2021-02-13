from queue import Queue
import gym
import matplotlib
import matplotlib.pyplot as plt
from models.util import record
import os

matplotlib.use('agg')


class RandomAgent:
    """Random Agent that will play the specified game
      Args:
        env_name: Name of the environment to be played
        max_eps: Maximum number of episodes to run agent for.
    """

    def __init__(self, env, max_eps=100, save_dir=None):
        self.env = env
        self.max_episodes = max_eps
        self.global_moving_average_reward = 0
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.res_queue = Queue()

    def train(self):
        reward_avg = 0
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
                reward_sum += reward

            # Record statistics
            moving_average_rewards.append(reward_sum)
            reward_avg += reward_sum
            self.global_moving_average_reward = record(
                episode, reward_sum, 0, self.global_moving_average_reward, self.res_queue, 0, steps)

        final_avg = reward_avg / float(self.max_episodes)
        print("Average score across {} episodes: {}".format(
            self.max_episodes, final_avg))

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average episode reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(self.save_dir,
                                 '{} Random Model Moving Average.png'.format(self.game_name)))
        return final_avg

    def play(self):
        state = self.env.reset()  # (84, 84, 3)
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                # self.env.render(mode='rgb_array')
                action = self.env.action_space.sample()
                obs, reward, done, info = self.env.step(action)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(
                    step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            return reward_sum
