import argparse

from models.random_agent import RandomAgent
from obstacle_tower_env import ObstacleTowerEnv
import time

from definitions import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run A3C algorithm on the Obstacle Tower Challenge game.')
    parser.add_argument('--env', default=None, type=str,
                        help='Path to OTC game executable.')
    parser.add_argument('--train', action='store_true',
                        default=False, help='Train the model.')
    parser.add_argument('--evaluate', action='store_true',
                        default=False, help='Evaluate the trained model.')
    parser.add_argument('--algorithm', default='a3c', type=str,
                        help='Choose between \'a3c\' and \'random\'.')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='Learning rate for the shared optimizer.')
    parser.add_argument('--max-eps', default=10, type=int,
                        help='Global maximum number of episodes to run.')
    parser.add_argument('--update-freq', default=20, type=int,
                        help='How often to update the global model.')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='Discount factor of rewards.')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of workers for asynchronous learning.')
    parser.add_argument('--rmsprop-alpha', default=0.99,
                        help='RMSProp decay factor.')
    parser.add_argument('--save-dir', default='./model_files/', type=str,
                        help='Directory in which you desire to save the model.')
    args = parser.parse_args()
    print(args)

    start_time = time.time()
    if args.algorithm == 'random':
        random_agent = RandomAgent(env_path=args.env, train=args.train,
                                   evaluate=args.evaluate, max_eps=args.max_eps, save_dir=args.save_dir)
        random_agent.train()
    elif args.algorithm == 'a3c':
        from models.a3c import MasterAgent
        master = MasterAgent(env_path=args.env, train=args.train, evaluate=args.evaluate, lr=args.lr, max_eps=args.max_eps,
                             update_freq=args.update_freq, gamma=args.gamma, num_workers=args.num_workers, save_dir=args.save_dir)
        master.build_graph().summary()
        master.train()
    elif args.algorithm == 'ppo':
        from models.a3c_ppo import MasterAgent
        master = MasterAgent(env_path=args.env, train=args.train, evaluate=args.evaluate, lr=args.lr, max_eps=args.max_eps,
                             update_freq=args.update_freq, gamma=args.gamma, num_workers=args.num_workers, save_dir=args.save_dir)
        master.build_graph().summary()
        master.train()
    else:
        print("Unsupported algorithm passed with --algorithm flag.")

    end_time = time.time()
    print("Program execution time = {} secs\n".format(end_time - start_time))
