import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils.np_utils import to_categorical
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class A2C():
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr, num_episodes, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model = model
        opt_actor = keras.optimizers.Adam(lr=lr)
        self.model.compile(loss=self.loss_func, optimizer=opt_actor)

        self.critic_model = critic_model
        opt_critic = keras.optimizers.Adam(lr=critic_lr)
        self.critic_model.compile(loss='mse', optimizer=opt_critic)

        self.num_episodes = num_episodes
        self.n = n
        self.x = np.array([])
        self.y = np.array([])
        self.error = np.array([])
        self.file_name = "output_" + str(time.time()) + "_" + str(self.n)
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

    def loss_func(self, target, pred):
        epsilon = 1e-7
        return -tf.reduce_mean(target * tf.log(pred + epsilon))

    def calculate_advantage(self, gamma, rewards, states):
        rewards = np.array(rewards) / 100.
        T = len(rewards)
        discounted_rewards = np.zeros((T,1))
        gammaN = gamma ** self.n

        values = self.critic_model.predict(np.array(states))

        for t in reversed(range(0, T)):
            cumulative_reward = 0.
            Vend = 0.
            if ((t + self.n) < T):
                Vend = values[t + self.n][0]
            for k in range(0, self.n):
                if (t + k < T):
                    cumulative_reward += (gamma ** k) * rewards[t + k]
                else:
                    break
            discounted_rewards[t] = gammaN * Vend + cumulative_reward
        return (values, discounted_rewards)

    def train(self, env, gamma=0.99):
        for episode in range(self.num_episodes):
            states, actions, rewards = self.generate_episode(env)
            values, discounted_rewards = self.calculate_advantage(gamma, rewards, states)
            advantages = discounted_rewards - values
            advantages = np.reshape(np.array(advantages), (-1,1))
            action_y_values = advantages * actions
            discounted_rewards = np.reshape(np.array(discounted_rewards), (-1,1))
            critic_y_values = discounted_rewards
            self.model.fit(np.array(states), np.array(action_y_values), verbose=0, batch_size=len(states))
            self.critic_model.fit(np.array(states), np.array(critic_y_values), verbose=0, batch_size=len(states))


            if (episode % 1000 == 0):
                print(str(episode))

            if (episode % 100 == 0):
                print(self.test(env, episode))

        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        return

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        states = []
        actions = []
        rewards = []
        state = env.reset()
        done = False
        while not done:
            states.append(state)
            reshaped_state = np.reshape(state, (1,-1))
            action_prob = (self.model.predict(reshaped_state))[0]
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            reshaped_action = to_categorical(action, num_classes=env.action_space.n)
            actions.append(reshaped_action)
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            state = next_state
        return states, actions, rewards

    def test(self, env, curr_episode):
        test_rewards = []
        for episode in range(100):
            states, action, rewards = self.generate_episode(env)
            test_rewards.append(np.sum(rewards))
        mean = np.mean(test_rewards)
        std = np.std(test_rewards)
        self.x = np.append(self.x, curr_episode)
        self.y = np.append(self.y, mean)
        self.error = np.append(self.error, std)

        if (curr_episode % 1000 == 0):
            np.save(self.file_name + "_x.npy", self.x)
            np.save(self.file_name + "_y.npy", self.y)
            np.save(self.file_name + "_error.npy", self.error)

            plt.errorbar(self.x, self.y, self.error, linestyle='None', marker='^')
            plt.xlabel("Episodes")
            plt.ylabel("Average Reward Per 100 Episodes")
            plt.savefig(self.file_name + '.png')
        return (mean, std)

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # define custom critic model
    state_size = env.observation_space.shape[0]
    critic_model = Sequential()
    critic_model.add(Dense(64, input_dim=state_size, activation='relu'))
    critic_model.add(Dense(64, activation='relu'))
    critic_model.add(Dense(32, activation='relu'))
    critic_model.add(Dense(1, activation='linear'))

    agent = A2C(model, lr, critic_model, critic_lr, num_episodes, n=50)
    agent.train(env)
    print("test result")
    print(agent.test(env))

    # TODO: Train the model using A2C and plot the learning curves.


if __name__ == '__main__':
    main(sys.argv)

