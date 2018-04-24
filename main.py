#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
import random
from algs.DQN import create_dqn
from algs.a2c import A2C
import time 
import logging 
import os.path

logger = logging.getLogger("10703")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')

logs_dir = "logs"
curr_model_dir = None 
models_dir = "models"

a2c_model = "algs/a2c.py"
dqn_model = "algs/DQN.py"

def parse_arguments():
	# Ex. python main.py --env CartPole-v0 --replay-batch 32 --model-name cartpole_dqn_w_mem --deepness deep
    parser = argparse.ArgumentParser(description='Experience Replay Argument Parser')
    parser.add_argument('--env-name',dest='env_name',type=str, default="LunarLander-v2")
    parser.add_argument('--alg', type=str, default="a2c", help="a2c or dqn")
    parser.add_argument('--model-name',dest='model_name',type=str,required=True)

    parser.add_argument('--seed', dest='seed', type=int)
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.5)
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--test-mod', type=int, default=500, help="Test every x number of episodes")

    # Utils
    parser.add_argument('--render',dest='render',action="store_true")
    parser.add_argument('--record-video-only',dest='record_video_only',action="store_true")
    parser.add_argument('--test-only', action="store_true")

    # A2C 
    parser.add_argument('--actor-model-path', type=str, default=None,
                        help="Path to the actor model file.")
    parser.add_argument('--actor-lr', type=float,
                        default=1e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-model-path', type=str, default=None)
    parser.add_argument('--critic-lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    
    parser.add_argument('--N', dest='N', type=int,
                        default=20, help="The number of steps in N-step A2C.")

    # DQN 
    parser.add_argument('--deepness',dest='deepness',type=str,default=False)
    parser.add_argument('--lr',dest='lr',type=int,default=0.0001)
    parser.add_argument('--alt-learn', dest='alt_learn', action="store_true")
    
    # ER
    parser.add_argument('--replay-batch',dest='replay_batch',type=int, default=32)
    parser.add_argument('--hindsight', action="store_true")
    parser.add_argument('--combined-replay', dest='combined_replay', action="store_true")
    parser.add_argument('--priority-replay', dest='priority_replay', action="store_true")
    # TODO: add diffeq ultimate replay evolution adaptation 

    return parser.parse_args()

def main(args):

    global curr_model_dir
    
    args = parse_arguments()

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    curr_model_dir = f"{models_dir}/{args.model_name}/"
    if not os.path.exists(curr_model_dir):
        os.makedirs(curr_model_dir)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    #start logging settings
    os.environ['TZ'] = 'EST+05EDT,M4.1.0,M10.5.0'
    time.tzset()

    
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
    log_file_name = f'{logs_dir}/{args.model_name}_{time_str}.log'
    hdlr = logging.FileHandler(log_file_name)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #end logging settings

    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session. 
    keras.backend.tensorflow_backend.set_session(sess)

    #assert(args.env in ["CartPole-v0", "MountainCar-v0", "LunarLander-v2"])
    assert(args.alg in ["a2c", "dqn"])

    logger.info(f"Command line args: {args}")
    logger.info(f"Log saving to {log_file_name}")
    logger.info(f"Alg: {args.alg}")
    

    if args.hindsight:
        assert(args.env_name == "MountainCar-v0")
        default_goal = np.array([[0.5]])
    else:
        default_goal = None 

    if args.seed != None:
        time_seed = args.seed
    else:
        #set consistent seed based on time
        time_seed = int(''.join(time_str.split('_'))) % (2 ** 32)
    
    np.random.seed(time_seed)
    logger.info(f"Numpy random seed {time_seed}")

    env = gym.make(args.env_name)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    print(f"Num env inputs (state space): {num_inputs}")
    print(f"Num env outputs (actions): {num_outputs}")

    if args.alg == "a2c":
        agent = A2C(env, args.model_name, args.actor_model_path, args.actor_lr, args.critic_model_path, args.critic_lr, N=args.N, logger=logger)
    elif args.alg == "dqn":
        agent, use_episodes, num_train_episodes, num_train_steps = create_dqn(logger, args, env, default_goal, curr_model_dir, time_seed)

    if args.record_video_only:
        agent.test(record_video=True)
        return 
    
    if args.test_only:
        agent.test()
    else:
        if args.alg == "a2c":
            agent.train(args.num_episodes, gamma=args.gamma, test_interval=args.test_mod, render=args.render)
        elif args.alg == "dqn":
            agent.train(use_episodes, num_train_episodes, num_train_steps, 
            False if args.replay_batch == 0 else args.replay_batch, 
            default_goal=default_goal)
        

    logger.info(f"Log saved to {log_file_name}")

if __name__ == '__main__':
	main(sys.argv)
