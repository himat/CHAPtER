#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Lambda
from keras import optimizers
import keras.backend as K
import random
from keras.models import load_model
from models.DQN import DQN_Agent
from replay.memory import Replay_Memory

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train_file',dest='train',type=int,default=1)
	parser.add_argument('--model_file',dest='model_file',type=str)
	parser.add_argument('--replay_memory',dest='replay_memory', type=int, default=0)
	parser.add_argument('--model_type',dest='model_type', type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env
	replay_memory = args.replay_memory
	model_type = args.model_type

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 
	agent = DQN_Agent(model_type, environment_name, replay_memory)
	agent.train()
	agent.test()

if __name__ == '__main__':
	main(sys.argv)
