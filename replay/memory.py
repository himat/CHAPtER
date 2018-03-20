#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
import random
from collections import deque
from replay.segment_tree import SumSegmentTree, MinSegmentTree

def encode_samples(sample_batch):
    return [(curr.copy(), reward, action_i, next_s.copy(), is_terminal) for curr, reward, action_i, next_s, is_terminal in sample_batch]

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

    def sample_batch(self, batch_size):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        samp = random.sample(self.experiences, batch_size)
        return encode_samples(samp)

    def append(self, transition):
        # Appends transition to the memory.     
        self.experiences.append(transition)
        assert(len(self.experiences) <= self.max_mem_size)
        self.mem_size = len(self.experiences)

# Based on OpenAI baselines implementation 
class Prioritized_Replay_Memory():

    def __init__(self, alpha, memory_size=50000, burn_in=10000):
        """
        alpha: float
            prioritization weight
            0: no prioritization (uniform sampling)
            1: full prioritization
        """

        self.experiences = []
        self.next_index = 0
        self.burn_in = burn_in
        self.mem_size = 0
        self.max_mem_size = memory_size
        self.alpha = alpha

        # Needs to be a power of 2
        tree_capacity = 1 
        while tree_capacity < self.max_mem_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.curr_max_priority = 1.0
        

    def _sample_proportional(self, batch_size):

        batch = []
        for _ in range(batch_size):

            # Uniformly sample an index
            index = None 
            while (index in batch or index is None): # Loop is only to ensure no repeats
                mass = random.random() * self.sum_tree.sum(0, len(self.experiences) - 1)

                index = self.sum_tree.find_prefixsum_idx(mass)

            batch.append(index)

        assert(len(batch) == batch_size)

        return batch 

    def sample_batch(self, batch_size, beta):
        assert(batch_size > 0 and beta > 0)

        sample_indexes = self._sample_proportional(batch_size)
        weights = []

        priority_total_sum = self.sum_tree.sum()
        N = len(self.experiences)

        # For normalization
        p_min = self.min_tree.min() / priority_total_sum
        max_weight = (p_min * N) ** (-beta)

        for index in sample_indexes:
            p_sample = self.sum_tree[index] / priority_total_sum

            weight = (p_sample * N) ** (-beta)
            normalized_weight = weight / max_weight
            weights.append(normalized_weight)

        sample_experiences = [self.experiences[i] for i in sample_indexes]

        return (encode_samples(sample_experiences), weights, sample_indexes)


    def append(self, transition):

        if len(self.experiences) < self.max_mem_size: # when the buffer hasn't reached capacity yet
            self.experiences.append(transition)
        else:
            self.experiences[self.next_index] = transition

        # Set to highest priority 
        self.sum_tree[self.next_index] = self.curr_max_priority ** self.alpha 
        self.min_tree[self.next_index] = self.curr_max_priority ** self.alpha 

        assert(len(self.experiences) <= self.max_mem_size)
        self.mem_size = len(self.experiences)

        self.next_index = (self.next_index + 1) % self.max_mem_size

    def update_priorities(self, indexes, new_priorities):
        assert(len(indexes) == len(new_priorities))

        for i, new_priority_i in zip(indexes, new_priorities):
            assert(new_priority_i > 0)
            assert(0 <= i < self.max_mem_size)

            self.sum_tree[i] = new_priority_i ** self.alpha 
            self.min_tree[i] = new_priority_i ** self.alpha 

            self.curr_max_priority = max(self.curr_max_priority, new_priority_i)




