from obstacle_tower_env import ObstacleTowerEnv
import sys
import argparse

def run_episode(env):
    done = False
    episode_reward = 0.0
    
    while not done:
        action = env.action_space.sample()

        # run one timestep of the environment's dynamics
        # obs - agent's observation of the current environment
        # reward - amount of reward returned from previous action
        # done - whether the episode has ended
        # info - auxialiary diagnostic information
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
    return episode_reward

def run_evaluation(env):
    while not env.done_grading():
        run_episode(env)
        env.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_filename', default='../../ObstacleTower/obstacletower', nargs='?')
    args = parser.parse_args()

    # Retro mode sets the visual observation to 84 * 84 and flattens the action space

    # Realtime mode determines whether the environment window will render the scene,
    # as well as whether the environment will run at realtime speed. Set this to `True`
    # to visual the agent behavior as you would in player mode.
    env = ObstacleTowerEnv(args.environment_filename, retro=False, realtime_mode=False)

    # set fixed seed for random number generator [0, 100)
    env.seed(10)

    # set a fixed floor number to start from subserquent resets
    env.floor(1)

    if env.is_grading():
        episode_reward = run_evaluation(env)
    else:
        while True:
            episode_reward = run_episode(env)
            print("Episode reward: " + str(episode_reward))
            env.reset()

    env.close()

