#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from random import sample
from collections import deque
import math



class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000, hindsight=False, default_goal=None):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        
        self.hindsight = hindsight
        self.experiences = deque(maxlen=memory_size)
        self.burn_in = burn_in
        self.max_mem_size = memory_size
        self.mem_size = 0
        self.default_goal = default_goal

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        return sample(self.experiences, batch_size)
        
    def append(self, transition):
        # Appends transition to the memory.
        curr_state, reward, action, next_state, is_terminal = transition     
        # if self.hindsight:
        #     curr_state = np.concatenate((curr_state, self.default_goal), axis=1)
        #     next_state = np.concatenate((next_state, self.default_goal), axis=1)
        self.experiences.append((curr_state, reward, action, next_state, is_terminal))
        assert(len(self.experiences) <= self.max_mem_size)

    def append_episode(self, episode):
        if self.hindsight:
            _, _, _, end_state, is_terminal = episode[-1]
            assert(is_terminal)
            for experience in episode:
                (curr_state, reward, action, next_state, is_terminal) = experience
                is_terminal = math.isclose(next_state[0, -1], end_state[0, -1], rel_tol=1e-6)
                curr_state = curr_state.copy()
                curr_state[0, -1] = end_state[0, -1]
                next_state = next_state.copy()
                next_state[0, -1] = end_state[0, -1]
                self.experiences.append((curr_state, reward, action, next_state, is_terminal))