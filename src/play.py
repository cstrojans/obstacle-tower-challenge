from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import sys
import argparse
from models.random import RandomAgent
from models.a3c import MasterAgent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="arguments for playing the OTC game")
    parser.add_argument('--env', default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--algorithm', type=str, default='a3c')
    parser.add_argument('--save-dir', default='./model_files/', type=str,
                        help='Directory from where you wish to load the model.')
    args = parser.parse_args()
    print(args)

    # instantiate the game environment
    env_reset_config = {
        "visual-theme": 0,
        "agent-perspective": 0
    }
    if args.eval:
        env = ObstacleTowerEnv(args.env, retro=True, realtime_mode=False, config=env_reset_config)
    else:
        env = ObstacleTowerEnv(args.env, retro=True, realtime_mode=True, config=env_reset_config)

    if args.algorithm == 'random':
        model = RandomAgent(env=env, save_dir=args.save_dir)
    elif args.algorithm == 'a3c':
        model = MasterAgent(env=env, save_dir=args.save_dir)

    if args.eval:
        eval_seeds = [1001, 1002, 1003, 1004, 1005]
        env = ObstacleTowerEvaluation(env, eval_seeds)

        # run episodes until evaluation is complete
        while not env.evaluation_complete:
            episode_reward = model.play()

        print(env.results)

    else:
        episode_reward = model.play()
        print("Episode reward: " + str(episode_reward))

    env.close()
