import argparse
from obstacle_tower_env import ObstacleTowerEnv
import sys
import time

from models.common.constants import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run different ML agents on the Obstacle Tower Challenge game.')
    parser.add_argument('--env', default=None, type=str, help='Path to OTC game executable.')
    
    subparsers = parser.add_subparsers(help='sub-command help', dest='subparser_name')

    # create the parser for the "random" agent
    parser_a = subparsers.add_parser('random', help='command line arguments for the Random agent')
    parser_a.add_argument('--max-eps', default=10, type=int, help='Maximum number of episodes (games) to run.')
    parser_a.add_argument('--save-dir', default='./model_files/', type=str, help='Directory in which you desire to save the model.')

    # create the parser for the "a3c" agent
    parser_b = subparsers.add_parser('a3c', help='command line arguments for the A3C agent')
    parser_b.add_argument('--lr', default=1e-4, type=float, help='Learning rate for the shared optimizer.')
    parser_b.add_argument('--max-eps', default=10, type=int, help='Maximum number of episodes (games) to run.')
    parser_b.add_argument('--update-freq', default=10000, type=int, help='How often to update the global model.')
    parser_b.add_argument('--gamma', default=0.99, type=float, help='Discount factor of rewards.')
    parser_b.add_argument('--num-workers', default=0, type=int, help='Number of workers for asynchronous learning.')
    parser_b.add_argument('--save-dir', default='./model_files/', type=str, help='Directory in which you desire to save the model.')

    # create the parser for the "curiosity" agent
    parser_c = subparsers.add_parser('curiosity', help='command line arguments for the Curiosity agent')
    parser_c.add_argument('--lr', default=1e-4, type=float, help='Learning rate for the shared optimizer.')
    parser_c.add_argument('--timesteps', default=10240, type=int, help='Maximum number of episodes (games) to run.')
    parser_c.add_argument('--batch-size', default=1024, type=int, help='How often to update the global model.')
    parser_c.add_argument('--gamma', default=0.99, type=float, help='Discount factor of rewards.')
    parser_c.add_argument('--save-dir', default='./model_files/', type=str, help='Directory in which you desire to save the model.')

    args, argv = parser.parse_known_args(sys.argv[1:])
    print(args)

    start_time = time.time()
    if args.subparser_name == 'random':
        from models.random.random_agent import RandomAgent
        agent = RandomAgent(env_path=args.env, train=True,
                                   evaluate=False, max_eps=args.max_eps, save_dir=args.save_dir)
        agent.train()
    elif args.subparser_name == 'a3c':
        from models.a3c.a3c_agent import MasterAgent
        agent = MasterAgent(env_path=args.env, train=True, evaluate=False, lr=args.lr, max_eps=args.max_eps,
                             update_freq=args.update_freq, gamma=args.gamma, num_workers=args.num_workers, save_dir=args.save_dir)
        # agent.build_graph().summary()
        agent.train()
    elif args.subparser_name == 'curiosity':
        from models.curiosity.curiosity_agent import CuriosityAgent
        agent = CuriosityAgent(env_path=args.env, train=True, evaluate=False, lr=args.lr, max_eps=args.max_eps,
                             update_freq=args.update_freq, gamma=args.gamma, num_workers=args.num_workers, save_dir=args.save_dir)
        # master.build_graph().summary()
        agent.train()
    else:
        print("Unsupported algorithm passed with --algorithm flag.")

    end_time = time.time()
    print("Program execution time = {} secs\n".format(end_time - start_time))
