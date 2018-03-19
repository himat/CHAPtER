#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Lambda
from keras import optimizers
import keras.backend as K
import random
from keras.models import load_model

class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		self.memory = list()
		self.memory_size = memory_size
		self.burn_in = burn_in

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		return random.sample(self.memory, min(batch_size, len(self.memory)))

	def append(self, transition):
		# Appends transition to the memory.
		self.memory.append(transition)
		if (len(self.memory) > self.memory_size):
			del self.memory[0]