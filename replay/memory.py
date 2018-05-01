#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
import random
from collections import deque
import math


from replay.segment_tree import SumSegmentTree, MinSegmentTree

def encode_samples(sample_batch):
    return [(curr.copy(), reward, action_i, next_s.copy(), is_terminal) for curr, reward, action_i, next_s, is_terminal in sample_batch]

class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000, combined=False, prioritized=False, hindsight=False, default_goal=None, priority_alpha=None):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        
        self.combined = combined
        self.prioritized = prioritized
        self.hindsight = hindsight

        # self.experiences = deque(maxlen=memory_size)
        self.experiences = []
        self.next_index = 0
        self.burn_in = burn_in
        self.mem_size = 0
        self.max_mem_size = memory_size

        ### PER
        if prioritized:
            assert(priority_alpha)
            self.alpha = priority_alpha
            # Needs to be a power of 2
            tree_capacity = 1 
            while tree_capacity < self.max_mem_size:
                tree_capacity *= 2

            self.sum_tree = SumSegmentTree(tree_capacity)
            self.min_tree = MinSegmentTree(tree_capacity)
            self.curr_max_priority = 1.0

        ### HER 
        self.default_goal = default_goal
        self.goal_dim = default_goal.shape[1] if default_goal is not None else None # How many dimensions the goal is defined over

    # Returns a batch_size set of experiences
    # WARNING: Replay_Memory.append() must be called prior to sample_batch when this is called in the training loop
    #   This is especially important when using combined replay
    def sample_batch(self, batch_size, beta=None):
        assert(batch_size > 0)

        ret_batch = None
        sample_weights = None 
        sample_indexes = None 

        if self.combined:
            batch_size -= 1 

        if self.prioritized:
            ret_batch, sample_weights, sample_indexes = self._priority_sample_batch(batch_size, beta)
        else: 
            # Uniform sampling 
            ret_batch = random.sample(self.experiences, batch_size)

        # Add latest experience to return sample
        if self.combined:

            # Maybe todo: fix so that you can't double sample the most recent experience

            # Assumes that the most recent experience was just added via the .append() call of this class
            latest_exp_index = (self.next_index - 1) % self.max_mem_size
            latest_exp = self.experiences[latest_exp_index]
            ret_batch.append(latest_exp) 

            if self.prioritized:
                sample_weights.append(1.0) # weight of 1 sounds good
                sample_indexes.append(latest_exp_index)

        return (encode_samples(ret_batch), sample_weights, sample_indexes)
      
    # Adding HER goal updates
    # Call this function and pass in the entire episode after episode is over
    def append_HER_episode(self, episode, reward_mod=None):
        
        if self.hindsight:
            _, _, _, end_state, is_terminal = episode[-1]
            assert(is_terminal)
            
            for experience in episode:
                (curr_state, reward, action, next_state, is_terminal) = experience

                is_terminal = np.allclose(next_state[0, 0:self.goal_dim], end_state[0, 0:self.goal_dim], rtol=1e-6)
                reward = reward_mod(experience) if reward_mod else reward
                # if is_terminal:
                #     print(f"Big terminal boys")
                curr_state = curr_state.copy()
                curr_state[0, -self.goal_dim:] = end_state[0, 0:self.goal_dim]
                next_state = next_state.copy()
                next_state[0, -self.goal_dim:] = end_state[0, 0:self.goal_dim]
                self.append((curr_state, reward, action, next_state, is_terminal))



    # Appends transition to the memory.     
    def append(self, transition):

        if len(self.experiences) < self.max_mem_size: # when the buffer hasn't reached capacity yet
            self.experiences.append(transition)
        else:
            self.experiences[self.next_index] = transition 

        assert(len(self.experiences) <= self.max_mem_size)
        self.mem_size = len(self.experiences)

        # Set to highest priority 
        if self.prioritized:
            self.sum_tree[self.next_index] = self.curr_max_priority ** self.alpha 
            self.min_tree[self.next_index] = self.curr_max_priority ** self.alpha 
        
        placed_index = self.next_index

        self.next_index = (self.next_index + 1) % self.max_mem_size

        return placed_index

    # Used for priority sampling: weighted sample from the cumulative sums
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

    def _priority_sample_batch(self, batch_size, beta):
        assert(batch_size > 0 and beta > 0)
        assert(self.prioritized)

        sample_weights = []
        sample_indexes = self._sample_proportional(batch_size)

        priority_total_sum = self.sum_tree.sum()
        N = len(self.experiences)

        # For normalization
        p_min = self.min_tree.min() / priority_total_sum

        with np.errstate(divide="raise"):
            max_weight = (p_min * N) ** (-beta)

        for index in sample_indexes:
            p_sample = self.sum_tree[index] / priority_total_sum

            weight = (p_sample * N) ** (-beta)
            normalized_weight = weight / max_weight
            sample_weights.append(normalized_weight)

        sample_experiences = [self.experiences[i] for i in sample_indexes]

        return (encode_samples(sample_experiences), sample_weights, sample_indexes)

    # Updating PER priorities 
    def update_priorities(self, indexes, new_priorities):
        assert(len(indexes) == len(new_priorities))
        assert(self.prioritized)

        for i, new_priority_i in zip(indexes, new_priorities):
            assert(new_priority_i > 0)
            assert(0 <= i < self.max_mem_size)

            self.sum_tree[i] = new_priority_i ** self.alpha 
            self.min_tree[i] = new_priority_i ** self.alpha 

            self.curr_max_priority = max(self.curr_max_priority, new_priority_i)

        