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
    parser_b.add_argument('--timesteps', default=10000, type=int, help='Maximum number of episodes (games) to run.')
    parser_b.add_argument('--batch-size', default=500, type=int, help='How often to update the global model.')
    parser_b.add_argument('--gamma', default=0.99, type=float, help='Discount factor of rewards.')
    parser_b.add_argument('--num-workers', default=0, type=int, help='Number of workers for asynchronous learning.')
    parser_b.add_argument('--save-dir', default='./model_files/', type=str, help='Directory in which you desire to save the model.')
    parser_b.add_argument('--distributed-train', default=False, action='store_true',
                          help='Use distributed tensorflow for faster and scalable training (not valid when algorithm flag is set to \'random\')')
    parser_b.add_argument('--plot', default=False, type=bool, help='Plot model results (rewards, loss, etc)')

    # create the parser for the "curiosity" agent
    parser_c = subparsers.add_parser('curiosity', help='command line arguments for the Curiosity agent')
    parser_c.add_argument('--lr', default=1e-4, type=float, help='Learning rate for the shared optimizer.')
    parser_c.add_argument('--timesteps', default=10000, type=int, help='Maximum number of episodes (games) to run.')
    parser_c.add_argument('--batch-size', default=500, type=int, help='How often to update the global model.')
    parser_c.add_argument('--gamma', default=0.99, type=float, help='Discount factor of rewards.')
    parser_c.add_argument('--save-dir', default='./model_files/', type=str, help='Directory in which you desire to save the model.')

    # Create the parser for the "ppo" agent
    parser_ppo = subparsers.add_parser('ppo', help='command line arguments for the PPO agent')
    parser_ppo.add_argument('--lr', default=1e-4, type=float, help='Learning rate for the shared optimizer.')
    parser_ppo.add_argument('--max-eps', default=10, type=int, help='Maximum number of episodes (games) to run.')
    parser_ppo.add_argument('--update-freq', default=10000, type=int, help='How often to update the global model.')
    parser_ppo.add_argument('--timesteps', default=10000, type=int, help='Maximum number of episodes (games) to run.')
    parser_ppo.add_argument('--batch-size', default=500, type=int, help='How often to update the global model.')
    parser_ppo.add_argument('--gamma', default=0.99, type=float, help='Discount factor of rewards.')
    parser_ppo.add_argument('--num-workers', default=0, type=int, help='Number of workers for asynchronous learning.')
    parser_ppo.add_argument('--save-dir', default='./model_files/', type=str, help='Directory in which you desire to save the model.')
    # parser_ppo.add_argument('--distributed-train', default=False, action='store_true',
    #                       help='Use distributed tensorflow for faster and scalable training (not valid when algorithm flag is set to \'random\')')
    parser_ppo.add_argument('--plot', default=False, type=bool, help='Plot model results (rewards, loss, etc)')
    args, argv = parser.parse_known_args(sys.argv[1:])
    print(args)

    start_time = time.time()
    if args.subparser_name == 'random':
        from models.random.random_agent import RandomAgent
        agent = RandomAgent(env_path=args.env, train=True,
                                   evaluate=False, max_eps=args.max_eps, save_dir=args.save_dir)
        agent.train()
    elif args.subparser_name == 'a3c':
        if args.distributed_train:
            from models.distributed_tf.distributed_agent import DistributedMasterAgent
            master = DistributedMasterAgent(env_path=args.env, train=True, evaluate=False, lr=args.lr, timesteps=args.timesteps,
                                            batch_size=args.batch_size, gamma=args.gamma, save_dir=args.save_dir, plot=args.plot)
            # master.build_graph().summary()
            master.distributed_train()
        else:
            from models.a3c.a3c_agent import MasterAgent
            agent = MasterAgent(env_path=args.env, train=True, evaluate=False, lr=args.lr, timesteps=args.timesteps,
                                batch_size=args.batch_size, gamma=args.gamma, num_workers=args.num_workers, save_dir=args.save_dir)
            # agent.build_graph().summary()
            agent.train()
    elif args.subparser_name == 'curiosity':
        from models.curiosity.curiosity_agent import CuriosityAgent
        agent = CuriosityAgent(env_path=args.env, train=True, evaluate=False, lr=args.lr, timesteps=args.timesteps,
                             batch_size=args.batch_size, gamma=args.gamma, save_dir=args.save_dir)
        # master.build_graph().summary()
        agent.train()
    elif args.subparser_name == 'ppo':
        from models.ppo.ppo_agent import MasterAgent
        master = MasterAgent(env_path=args.env, train=True, evaluate=False, lr=args.lr, timesteps=args.timesteps,
                                batch_size=args.batch_size, gamma=args.gamma, num_workers=args.num_workers, save_dir=args.save_dir)
        # master.build_graph().summary()
        master.train()
    else:
        print("Unsupported algorithm passed.")

    end_time = time.time()
    print("Program execution time = {} secs\n".format(end_time - start_time))
