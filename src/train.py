import argparse
from models.a3c import MasterAgent
from models.random import RandomAgent
import multiprocessing
from obstacle_tower_env import ObstacleTowerEnv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run A3C algorithm on the Obstacle Tower Challenge game.')
    parser.add_argument('--env', default=None)
    parser.add_argument('--algorithm', default='a3c', type=str, help='Choose between \'a3c\' and \'random\'.')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate for the shared optimizer.')
    parser.add_argument('--max-eps', default=10, type=int, help='Global maximum number of episodes to run.')
    parser.add_argument('--update-freq', default=20, type=int, help='How often to update the global model.')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor of rewards.')
    parser.add_argument('--num-workers', default=multiprocessing.cpu_count(), type=int, help='Number of workers for asynchronous learning.')
    parser.add_argument('--rmsprop-alpha', default=0.99, help='RMSProp decay factor.')
    parser.add_argument('--save-dir', default='./model_files/', type=str, help='Directory in which you desire to save the model.')
    args = parser.parse_args()
    print(args)

    # instantiate the game environment
    env_reset_config = {
        "tower-seed": 99,  # fix floor generation seed to remove generalization
        "visual-theme": 0  # default theme to remove generalization while training
    }
    env = ObstacleTowerEnv(args.env, retro=True, realtime_mode=False, config=env_reset_config)

    if args.algorithm == 'random':
        random_agent = RandomAgent(env, args.max_eps, args.save_dir)
        random_agent.train()
    elif args.algorithm == 'a3c':
        master = MasterAgent(env, args.lr, args.max_eps, args.update_freq, args.gamma, args.num_workers, args.save_dir)
        master.build_graph().summary()
        master.train()
        
    else:
        print("Unsupported algorithm passed with --algorithm flag.")
    
    env.close()
