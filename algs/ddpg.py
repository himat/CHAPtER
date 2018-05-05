""" 
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf
The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow
Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp

import statistics
import logging
logger = logging.getLogger('10703')

from replay.memory import Replay_Memory

EPS = 1e-6

# ===========================
#   Actor and Critic DNNs
# ===========================
class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


class DDPG():
    def __init__(self, env, critic_lr=0.0001, actor_lr=0.0001, gamma=0.99, tau=0.001, batch_size=64, default_goal=None):
        action_dim = env.action_space.shape[0]

        self.sess = tf.Session()
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        state_dim = env.observation_space.shape[0]
        action_bound = env.action_space.high
        action_lower_bound = env.action_space.low
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low).all()
        if default_goal is not None:
            state_dim += default_goal.shape[1]

        self.actor = ActorNetwork(self.sess, state_dim, action_dim, action_bound,
                             actor_lr, tau,
                             batch_size)

        self.critic = CriticNetwork(self.sess, state_dim, action_dim,
                               critic_lr, tau,
                               gamma,
                               self.actor.get_num_trainable_vars())
        
        
# ===========================
#   Agent Training
# ===========================

    def burn_in(self, env, buf, exp_limit=10000, default_goal=None):
        logger.info(f"Burning in {exp_limit} experiences")

        steps = 0
        while steps < exp_limit:
            terminal = False

            s = env.reset().reshape((1, -1))
            if default_goal is not None:
                s = np.concatenate([s, default_goal], axis=1)
            current_episode = []
            while not terminal:                
                a = self.actor.predict(s) + self.actor_noise()
                a = np.clip(a, env.action_space.low, env.action_space.high) # Make sure outputs are valid
                
                s2, r, terminal, info = env.step(a[0])
                s2 = s2.reshape((1,-1))

                if default_goal is not None:
                    s2 = np.concatenate([s2, default_goal], axis=1)
            
                exp = (s, r, a, s2, terminal)
                current_episode.append(exp)
                buf.append(exp) # Add exps with original goals
                s = s2
            steps += len(current_episode)

            if buf.hindsight:
                buf.append_HER_episode(current_episode) # Add exps with HER goals

    def train(self, env, num_eps=10000, combined_replay=False, hindsight_replay=False, 
        priority_replay=False, priority_replay_alpha=0.6, render=False, default_goal=None, batch_size=64, train_mod=1,
        test_mod=1000):

        priority_replay_alpha = 0.6
        priority_replay_beta_init = 0.4
        priority_replay_beta = priority_replay_beta_init # TODO: anneal
        
        sess = self.sess
        # Set up summary Ops
        summary_ops, summary_vars = build_summaries()

        sess.run(tf.global_variables_initializer())
        #writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()

        # Initialize replay memory
        replay_buffer = Replay_Memory(combined=combined_replay, prioritized=priority_replay, 
            hindsight=hindsight_replay, default_goal=default_goal, priority_alpha=priority_replay_alpha)

        self.burn_in(env, replay_buffer, default_goal=default_goal)

        # Needed to enable BatchNorm. 
        # This hurts the performance on Pendulum but could be useful
        # in other environments.
        # tflearn.is_training(True)

        train_reward = []
        best_test_reward = None
        best_test_epoch = None
       
        for eps_idx in range(num_eps):

            s = env.reset().reshape((1, -1))
            
            if default_goal is not None:
                s = np.concatenate([s, default_goal], axis=1)

            terminal = False

            current_episode = []
            current_reward = 0

            if eps_idx % train_mod == 0 or eps_idx % test_mod == 0:
                logger.info(f"Episode {eps_idx}")

            while not terminal:

                if render:
                    env.render()
                # Added exploration noise
                #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
                a = self.actor.predict(s) + self.actor_noise()
                a = np.clip(a, env.action_space.low, env.action_space.high) # Make sure outputs are valid

                s2, r, terminal, info = env.step(a[0])
                s2 = s2.reshape((1, -1))
                if default_goal is not None:
                    s2 = np.concatenate([s2, default_goal], axis=1)
                # print(f"Reward {r}")
                # if r > 50.0:
                #     print(f"Big reward of {r}")
                current_reward += r

                exp = (s, r, a, s2, terminal)
                current_episode.append(exp)
                replay_buffer.append(exp)
               
                samples, sample_weights, sample_indexes = replay_buffer.sample_batch(batch_size, beta=priority_replay_beta)

                s_batch, r_batch, a_batch, s2_batch, t_batch = \
                    tuple([np.concatenate(list(map(lambda x: x[i], samples))) if i == 0 or i == 3 or i == 2 else 
                        np.array(list(map(lambda x: x[i], samples))) for i in range(5)])

                # Calculate target_scaled_out
                target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

                y_i = []
                for k in range(batch_size):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + self.critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = self.critic.train(
                    s_batch, a_batch, np.reshape(y_i, (-1, 1)))


                # Update the actor policy using the sampled gradient
                a_outs = self.actor.predict(s_batch)
                grads = self.critic.action_gradients(s_batch, a_outs)
                self.actor.train(s_batch, grads[0])

                # Update target networks
                self.actor.update_target_network()
                self.critic.update_target_network()

                s = s2


                if priority_replay:
                    
                    td_errors = np.reshape(y_i, (-1, 1)) - predicted_q_value 
                    replay_buffer.update_priorities(sample_indexes, np.abs(td_errors) + EPS)

            if replay_buffer.hindsight:
                replay_buffer.append_HER_episode(current_episode, 
                    reward_mod=lambda x: 100 if x[4] else x[1])

            train_reward.append(current_reward)

            if eps_idx % train_mod == 0 and eps_idx != 0:
                logger.info("Episode reward mean: {:.3f}".format(statistics.mean(train_reward)))
                logger.info("Episode reward std: {:.3f}".format(statistics.stdev(train_reward)))
                train_reward = []

            if eps_idx % test_mod == 0 and eps_idx != 0:
                test_reward_list = self.test(env, num_eps=100, default_goal=default_goal)
                test_reward_avg = statistics.mean(test_reward_list)
                test_reward_stdev = statistics.stdev(test_reward_list)
                logger.info("(test) Test reward mean: {:.3f}".format(test_reward_avg))
                logger.info("Test reward std: {:.3f}".format(test_reward_stdev))

                if best_test_reward == None or test_reward_avg > best_test_reward:
                    logger.info(f"New best reward of {test_reward_avg}")
                    logger.info(f"Beat old reward of {best_test_reward}")
                    best_test_reward = test_reward_avg
                    best_test_epoch = eps_idx

                    # TODO: save model here

        logger.info(f"Best test reward: {best_test_reward} at epoch {best_test_epoch}")
                

    def test(self, env, num_eps=100, default_goal=None):
        logger.info("**Testing mode")
        rewards = []
        for i in range(num_eps):
            terminal = False

            s = env.reset().reshape((1, -1))
            reward = 0
            while not terminal:
                if default_goal is not None:
                    s = np.concatenate([s, default_goal], axis=1) 

                a = self.actor.predict(s) + self.actor_noise()
                a = np.clip(a, env.action_space.low, env.action_space.high) # Make sure outputs are valid

                s2, r, terminal, info = env.step(a[0])
                reward += r
                s = s2.reshape((1, -1))
            rewards.append(reward)
        return rewards