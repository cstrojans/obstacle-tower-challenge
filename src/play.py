from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import sys
import argparse
from models.random_agent import RandomAgent
from models.a3c import MasterAgent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="arguments for playing the OTC game")
    parser.add_argument('--env', default=None, type=str,
                        help='Path to OTC game executable.')
    parser.add_argument('--evaluate', action='store_true',
                        default=False, help='Evaluate the trained model.')
    parser.add_argument('--algorithm', type=str, default='a3c',
                        help='Choose between \'a3c\' and \'random\'.')
    parser.add_argument('--save-dir', default='./model_files/', type=str,
                        help='Directory from where you wish to load the model.')
    args = parser.parse_args()
    print(args)

    eval_seeds = [1001, 1002, 1003, 1004, 1005]
    if args.algorithm == 'random':
        model = RandomAgent(env_path=args.env, train=False,
                            evaluate=args.evaluate, eval_seeds=eval_seeds, save_dir=args.save_dir)
    elif args.algorithm == 'a3c':
        model = MasterAgent(env_path=args.env, train=False, evaluate=args.evaluate, eval_seeds=eval_seeds, lr=0.0,
                            max_eps=0, update_freq=0, gamma=0, num_workers=1, save_dir=args.save_dir)

    if args.evaluate:  # perform evaluation
        model.evaluate()
    else:
        episode_reward = model.play_single_episode()
        print("Episode reward: " + str(episode_reward))
