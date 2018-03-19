#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Lambda
from keras import optimizers
import keras.backend as K
import random
from keras.models import load_model
from replay.memory import Replay_Memory

class Q_Network():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, model_type, environment_name):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
		self.environment_name = environment_name
		self.env = gym.make(environment_name)
		self.state_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n
		
		if model_type == "duel": 
			x = Input(shape=(self.state_size,))
			h = x
			for i in range(3):
				h = Dense(24, activation='tanh')(h)
			y = Dense(self.action_size+1)(h)
			z = Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:], output_shape=(self.action_size,))(y)
			self.model = Model(inputs=x, outputs=z)

		elif model_type == "dqn":
			self.model = Sequential()
			self.model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
			self.model.add(Dense(24, activation='tanh'))
			self.model.add(Dense(self.action_size, activation='linear'))

		elif model_type == "linear":
			self.model = Sequential()
			self.model.add(Dense(self.action_size, kernel_initializer='random_normal', input_dim=self.state_size))

		opt = optimizers.Adam(lr=0.1, decay=1e-6)#, clipnorm=1.)
		self.model.compile(loss='mse', optimizer=opt)

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights. 
		self.model.save_weights('model_weights' + suffix + '.h5')

	def load_model(self, model_file):
		# Helper function to load an existing model.
		self.model = load_model(model_file)

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		self.model.load_weights(weight_file)

class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, model_type, environment_name, replay_memory, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		self.gamma = 1 if environment_name == "MountainCar-v0" else 0.99
		self.epsilon_max = 0.5
		self.epsilon_min = 0.05
		self.episodes_max = 5000 if environment_name == "MountainCar-v0" else 1000000
		self.batch_size = 32

		self.environment_name = environment_name
		self.model_type = model_type
		self.render = render
		self.train_env = gym.make(self.environment_name)
		self.test_env = gym.make(self.environment_name)
		self.Q_network = Q_Network(model_type, environment_name)
		self.replay_memory = replay_memory

		self.iter = 0

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.             
		random_number = np.random.rand(1)[0]
		offset = (self.epsilon_max - self.epsilon_min) / 1000000 * self.iter
		if random_number < max(self.epsilon_max - offset, self.epsilon_min):
			return np.random.randint(len(q_values)) # end points included
		else: 
			return np.argmax(q_values)

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		return np.argmax(q_values)

	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		
		if not self.replay_memory:
			print("not replay memory")
			best_reward = 0.
			for episode in range(self.episodes_max):
				state = self.train_env.reset()
				done = False
				while not done:
					if self.render:
						env.render()
					reshaped_state = np.reshape(state,(1,self.Q_network.state_size))
					q_values = self.Q_network.model.predict(reshaped_state)
					action = self.epsilon_greedy_policy(q_values)
					next_state, reward, done, info = self.train_env.step(action)
					state = next_state
					self.iter += 1

					reshaped_next_state = np.reshape(state,(1,self.Q_network.state_size))
					target = self.Q_network.model.predict(reshaped_state)
					if done:
						target[0][action] = reward
					else:
						next_q_values = self.Q_network.model.predict(reshaped_next_state)
						target[0][action] = reward + self.gamma * np.amax(next_q_values)
					self.Q_network.model.fit(reshaped_state, target, verbose=0)
					

					if (self.iter % 10000 == 0):
						current_reward = self.test()
						print(str(current_reward) + " iter " + str(self.iter) + " epi " + str(episode))
						if (current_reward > best_reward):
							best_reward = current_reward
						file_name = 'model_' + str(self.iter) + '.h5'
						print("Saving model...")
						self.Q_network.model.save(file_name)

				if self.environment_name == "CartPole-v0" and self.iter > 1000000: 
					return
			return

		else:
			print("replay memory")
			replay_memory = Replay_Memory()
			best_reward = 0.
			stuck = 0.
			for episode in range(self.episodes_max):
				state = self.train_env.reset()
				done = False
				while not done:
					if self.render:
						env.render()
					reshaped_state = np.reshape(state, (1,self.Q_network.state_size))
					q_values = self.Q_network.model.predict(reshaped_state)
					action = self.epsilon_greedy_policy(q_values)
					next_state, reward, done, info = self.train_env.step(action)
					replay_memory.append((state, action, reward, next_state, done))
					state = next_state
					self.iter += 1
				
					minibatch = replay_memory.sample_batch(self.batch_size)
					inputs  = np.array([x[0] for x in minibatch])
					targets = np.array(self.Q_network.model.predict(inputs))
					next_states = np.array([x[3] for x in minibatch])
					next_q_values = np.array(self.Q_network.model.predict(next_states))
					for i in range(len(minibatch)):
						mini_state, mini_action, mini_reward, mini_next_state, mini_done = minibatch[i]
						if mini_done or (self.environment_name == "MountainCar-v0" and self.model_type == "linear" and next_states[0] > 0.5):
							targets[i][mini_action] = mini_reward
						else:
							targets[i][mini_action] = reward + self.gamma * np.amax(next_q_values[i])
					self.Q_network.model.fit(inputs, targets, verbose=0)

					if (self.iter % 1000 == 0):
						current_reward = self.test()
						print(str(current_reward) + " iter " + str(self.iter) + " epi " + str(episode))
						if (current_reward > best_reward):
							best_reward = current_reward
						file_name = 'model_' + str(self.iter) + '.h5'
						print("Saving model...")
						self.Q_network.model.save(file_name)
				if self.environment_name == "CartPole-v0" and self.iter > 1000000: 
					return
			return
	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		if model_file != None:
			model = load_model(model_file)
		else:
			model = self.Q_network.model
		total_reward = 0.
		episodes_max = 20
		self.epsilon_max = 0.05
		for episode in range(episodes_max):
			test_done = False
			state = self.test_env.reset()

			while not test_done:
				state = np.reshape(state,(1,self.Q_network.state_size))
				q_values = self.Q_network.model.predict(state)
				action = self.epsilon_greedy_policy(q_values)
				test_next_state, test_reward, test_done, test_info = self.test_env.step(action)
				total_reward += test_reward
				state = test_next_state

		total_reward /= episodes_max
		return total_reward



	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions. 
		return Replay_Memory()