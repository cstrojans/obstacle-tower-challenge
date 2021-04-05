from prettyprinter import pprint
import os
import time
from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
from models.common.constants import train_env_reset_config, eval_env_reset_config
from models.common.util import instantiate_environment
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


class StablePPO():
    def __init__(self, env_path, train, evaluate, policy_name='CnnPolicy', save_dir='./model_files/', eval_seeds=[]):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.model_path = os.path.join(self.save_dir, 'model_stable_ppo')
        self.log_dir = './logs/stable_ppo'
        self.policy_name = policy_name
        self.evaluate = evaluate

        if train:
            self.env = ObstacleTowerEnv(
                env_path, worker_id=0, retro=True, realtime_mode=False, config=train_env_reset_config)
        else:
            if evaluate:
                self.env = ObstacleTowerEnv(
                    env_path, worker_id=0, retro=True, realtime_mode=False, config=eval_env_reset_config)
                self.env = ObstacleTowerEvaluation(self.env, eval_seeds)
            else:
                self.env = ObstacleTowerEnv(
                    env_path, worker_id=0, retro=True, realtime_mode=True, config=eval_env_reset_config)

    def load_model(self):
        print('Loading model from: {}'.format(self.model_path))
        model = PPO.load(self.model_path)
        model.set_env(self.env)
        model.tensorboard_log = self.log_dir
        return model

    def train(self, timesteps=10000, continue_training=True):
        start_time = time.time()
        if continue_training:
            model = PPO(self.policy_name, self.env, verbose=1,
                        tensorboard_log=self.log_dir)
        else:
            model = self.load_model()

        model.learn(total_timesteps=timesteps)
        print('\nTraining complete. Time taken = {} secs'.format(time.time() - start_time))
        model.save(self.model_path)

    def play_single_episode(self):
        """ have the trained agent play a single game """
        model = self.load_model()

        print("Playing single episode...")
        done = False
        reward_sum = 0
        step_counter = 0
        obs = self.env.reset()
        try:
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                print("{}. Reward: {}, action: {}".format(
                    step_counter, reward_sum, action))
                self.env.render()
                step_counter += 1
                reward_sum += reward
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
