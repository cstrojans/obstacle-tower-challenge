from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import sys
import argparse
from models.random import RandomPolicy

def run_episode(env, agent_brain):
    done = False
    episode_reward = 0.0
    
    obs = env.reset()
    while not done:
        # TODO: get next action from agent's policy
        action = agent_brain.predict(obs)

        # perform the action on the environment
        obs, reward, done, info = env.step(action)

        # update cumulative reward
        episode_reward += reward
        
    return episode_reward

def run_evaluation(env):
    while not env.done_grading():
        run_episode(env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="arguments for playing the OTC game")
    parser.add_argument('-env', default=None)
    parser.add_argument('-eval', action='store_true')
    parser.add_argument('-policy', type=str, default='random')
    args = parser.parse_args()

    # instantiate the game environment
    env = ObstacleTowerEnv(args.env, retro=False, realtime_mode=False)

    # TODO: load trained model based on policy
    if args.policy == 'random':
        agent_brain = RandomPolicy(env)

    if args.eval:
        eval_seeds = [1001]
        env = ObstacleTowerEvaluation(env, eval_seeds)

        # run episodes until evaluation is complete
        while not env.evaluation_complete:
            episode_rew = run_episode(env, agent_brain)

        print(env.results)

    else:
        episode_reward = run_episode(env, agent_brain)
        print("Episode reward: " + str(episode_reward))

    env.close()
