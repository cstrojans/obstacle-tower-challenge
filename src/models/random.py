from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import sys
import argparse

class RandomPolicy:
    def __init__(self, env):
        self.env = env
    
    def create_model(self):
        pass

    def train(self):
        pass

    def predict(self, obs):
        return self.env.action_space.sample()
