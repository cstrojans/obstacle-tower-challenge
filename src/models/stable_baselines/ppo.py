from prettyprinter import pprint
import os
import time
from models.common.util import ActionSpace
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


class StablePPO():
    def __init__(self, env_path, train, evaluate, policy_name='CnnPolicy', save_dir='./model_files/', eval_seeds=[], reduced_action=False):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.model_path = os.path.join(self.save_dir, 'model_stable_ppo')
        self.log_dir = './logs/stable_ppo'
        self.policy_name = policy_name
        self.evaluate = evaluate
        print(env_path)
        if reduced_action:
            from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
            from models.common.constants import train_env_reset_config_industrial as train_env_reset_config
            from models.common.constants import eval_env_reset_config_industrial as eval_env_reset_config
        else:
            from models.stable_baselines.reduced_action_env import ObstacleTowerEnv, ObstacleTowerEvaluation
            from models.common.constants import train_env_reset_config, eval_env_reset_config
        
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

    def train(self, timesteps=10000, continue_training=False):
        start_time = time.time()
        if not continue_training:
            print("Initializing from scratch")
            model = PPO(self.policy_name, self.env, verbose=1,
                        tensorboard_log=self.log_dir, n_steps=6144, n_epochs=4)
        else:
            model = self.load_model()
            print("Restored from {}".format(self.model_path))

        model.learn(total_timesteps=timesteps)
        print('\nTraining complete. Time taken = {} secs'.format(time.time() - start_time))
        model.save(self.model_path)

    def play_single_episode(self):
        """ have the trained agent play a single game """
        action_space = ActionSpace()
        done = False
        reward_sum = 0
        step_counter = 0

        model = self.load_model()
        obs = self.env.reset()
        try:
            print("Playing single episode...")
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                print("{}. Reward: {}, action: {}".format(
                    step_counter, reward_sum, action_space.get_full_action_meaning(action)))
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
