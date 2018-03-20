#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from random import sample
from collections import deque

class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        
        self.experiences = deque(maxlen=memory_size)
        self.burn_in = burn_in
        self.max_mem_size = memory_size
        self.mem_size = 0

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        samp = sample(self.experiences, batch_size)
        return [(curr.copy(), reward, action_i, next_s.copy(), is_terminal) for curr, reward, action_i, next_s, is_terminal in samp]

    def append(self, transition):
        # Appends transition to the memory.     
        self.experiences.append(transition)
        assert(len(self.experiences) <= self.max_mem_size)