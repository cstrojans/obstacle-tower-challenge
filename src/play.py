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
                        help='Choose between \'random\', \'a3c\', \'stable_a2c\', \'stable_ppo\', and \'curiosity\'.')
    parser.add_argument('--save-dir', default='./model_files/', type=str,
                        help='Directory from where you wish to load the model.')
    parser.add_argument('--reduced-action', default=False, action='store_true', help='Use a reduced set of actions for training')
    args = parser.parse_args()
    print(args)

    # eval_seeds = [1001, 1002, 1003, 1004, 1005]
    eval_seeds = [1001]
    if args.algorithm == 'random':
        from models.random.random_agent import RandomAgent
        agent = RandomAgent(env_path=args.env, train=False,
                            evaluate=args.evaluate, eval_seeds=eval_seeds, save_dir=args.save_dir)
    elif args.algorithm == 'a3c':
        from models.a3c.a3c_agent import MasterAgent
        agent = MasterAgent(env_path=args.env, train=False, evaluate=args.evaluate, eval_seeds=eval_seeds, lr=0.0,
                            timesteps=0, batch_size=0, gamma=0, num_workers=1, save_dir=args.save_dir)
    elif args.algorithm == 'a3c_distributed':
        from models.distributed_tf.distributed_agent import DistributedMasterAgent
        agent = DistributedMasterAgent(env_path=args.env, train=False, evaluate=args.evaluate, lr=0.0, timesteps=0, batch_size=0, gamma=0, save_dir=args.save_dir, plot=False)
    elif args.algorithm == 'curiosity':
        from models.curiosity.curiosity_agent import CuriosityAgent
        agent = CuriosityAgent(env_path=args.env, train=False, evaluate=args.evaluate,
                               eval_seeds=eval_seeds, lr=0.0, timesteps=0, batch_size=0, gamma=0, save_dir=args.save_dir)
    
    elif args.algorithm == 'stable_a2c':
        from models.stable_baselines.a2c import StableA2C
        agent = StableA2C(env_path=args.env, train=False, evaluate=args.evaluate, policy_name='', save_dir=args.save_dir, eval_seeds=eval_seeds)
    
    elif args.algorithm == 'stable_ppo':
        from models.stable_baselines.ppo import StablePPO
        agent = StablePPO(env_path=args.env, train=False, evaluate=args.evaluate, policy_name='', save_dir=args.save_dir, eval_seeds=eval_seeds, reduced_action=args.reduced_action)
    
    elif args.algorithm == 'ppo':
        from models.ppo.ppo_agent import MasterAgent
        agent = MasterAgent(env_path=args.env, train=False, evaluate=args.evaluate, eval_seeds=eval_seeds, lr=0.0,
                            timesteps = 1000, batch_size=1024, gamma=0, num_workers=1, save_dir=args.save_dir)
    
    else:
        print("Unsupported algorithm passed with --algorithm flag.")
    
    if args.evaluate:  # perform evaluation
        agent.evaluate()
    else:
        episode_reward = agent.play_single_episode()
        print("Episode reward: " + str(episode_reward))
