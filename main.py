#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
import random
from models.DQN import DQN_Agent
import time 
import logging 
import os.path

logger = logging.getLogger("10703")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')

logs_dir = "logs"
curr_model_dir = None 
models_dir = "models"

def parse_arguments():
	# Ex. python main.py --env CartPole-v0 --replay-batch 32 --model-name cartpole_dqn_w_mem --deepness deep
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str,required=True)
    parser.add_argument('--model-name',dest='model_name',type=str,required=True)
    parser.add_argument('--render',dest='render',action="store_true")
    parser.add_argument('--test-only',dest='test_only',action="store_true")
    parser.add_argument('--record-video-only',dest='record_video_only',action="store_true")
    parser.add_argument('--replay-batch',dest='replay_batch',type=int, default=32)
    parser.add_argument('--deepness',dest='deepness',type=str,default=False)
    parser.add_argument('--seed', dest='seed', type=int)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001)
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.5)
    parser.add_argument('--alt-learn', dest='alt_learn', action="store_true")
    parser.add_argument('--num-eps', dest='num_eps', type=int, default=None)
    
    parser.add_argument('--hindsight', action="store_true")
    parser.add_argument('--combined-replay', dest='combined_replay', action="store_true")
    parser.add_argument('--priority-replay', dest='priority_replay', action="store_true")

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

    logger.info(f"{args}")


    env_name = args.env

    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session. 
    keras.backend.tensorflow_backend.set_session(sess)

    num_train_episodes = args.num_eps

    if env_name == "CartPole-v0":
        gamma = 1.0
        use_episodes = False
        if not args.num_eps:
            num_train_episodes = None
        num_train_steps = 800000
         
    elif env_name == "MountainCar-v0":
        gamma = 0.99
        use_episodes = True 
        if not args.num_eps:
            num_train_episodes = 8000 #5000
        num_train_steps = None

    else:
        raise Exception("Invalid env")

    if args.deepness:
        assert(args.deepness == "deep" or args.deepness == "dueling")

    # You want to create an instance of the DQN_Agent class here, and then train / test it.


    default_goal = np.array([[0.5]])

    if args.seed != None:
        time_seed = args.seed
    else:
        #set consistent seed based on time
        time_seed = int(''.join(time_str.split('_'))) % (2 ** 32)
    
    np.random.seed(time_seed)
    logger.info(f"Numpy random seed {time_seed}")

    agent = DQN_Agent(curr_model_dir, logger, env_name, gamma, eps_init=args.epsilon, lr_init=args.lr, 
        render=args.render, test_mode=args.test_only, model_name=args.model_name, 
        deep=args.deepness, seed=time_seed, alt_learn=args.alt_learn, 
        goal_size=default_goal.shape[1] if args.hindsight else 0,
        combined_replay=args.combined_replay, priority_replay=args.priority_replay)

    if args.record_video_only:
        agent.test(record_video=True)
        return 
    
    if not args.test_only:
        agent.train(use_episodes, num_train_episodes, num_train_steps, 
            False if args.replay_batch == 0 else args.replay_batch, 
            hindsight=args.hindsight, default_goal=default_goal)
    else:
        agent.test()

    logger.info(f"Log saved to {log_file_name}")

if __name__ == '__main__':
	main(sys.argv)
