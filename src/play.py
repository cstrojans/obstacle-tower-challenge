from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
import sys
import argparse


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
        from models.random.random_agent import RandomAgent
        agent = RandomAgent(env_path=args.env, train=False,
                            evaluate=args.evaluate, eval_seeds=eval_seeds, save_dir=args.save_dir)
    elif args.algorithm == 'a3c':
        from models.a3c.a3c_agent import MasterAgent
        agent = MasterAgent(env_path=args.env, train=False, evaluate=args.evaluate, eval_seeds=eval_seeds, lr=0.0,
                            max_eps=0, update_freq=0, gamma=0, num_workers=1, save_dir=args.save_dir)
    elif args.algorithm == 'curiosity':
        from models.curiosity.curiosity_agent import CuriosityAgent
        agent = CuriosityAgent(env_path=args.env, train=False, evaluate=args.evaluate, lr=0, timesteps=0,
                             batch_size=0, gamma=0, save_dir=args.save_dir)
    else:
        print("Unsupported algorithm passed with --algorithm flag.")
    
    if args.evaluate:  # perform evaluation
        agent.evaluate()
    else:
        episode_reward = agent.play_single_episode()
        print("Episode reward: " + str(episode_reward))
