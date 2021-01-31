from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import sys
import argparse

def run_episode(env):
    """run_episode runs a game by executing a random policy

    Parameters
    ----------
    env : ObstacleTowerEnv
        game environment with OpenAI gym wrapper

    Returns
    -------
    float
        reward earned in the episode
    """
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
    parser = argparse.ArgumentParser(description="arguments for playing the OTC game")
    parser.add_argument('-env', default=None)
    parser.add_argument('-eval', action='store_true')
    args = parser.parse_args()

    # Retro mode sets the visual observation to 84 * 84 and flattens the action space

    # Realtime mode determines whether the environment window will render the scene,
    # as well as whether the environment will run at realtime speed. Set this to `True`
    # to visual the agent behavior as you would in player mode.
    env = ObstacleTowerEnv(args.env, retro=False, realtime_mode=False)

    if args.eval:
        eval_seeds = [1001]
        env.reset()
        env = ObstacleTowerEvaluation(env, eval_seeds)

        # run episodes until evaluation is complete
        while not env.evaluation_complete:
            episode_rew = run_episode(env)

        print(env.results)
        env.close()

    else:
        episode_reward = run_episode(env)
        print("Episode reward: " + str(episode_reward))

        env.close()
