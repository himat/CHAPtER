import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce

from keras.models import Sequential, Model
from keras.layers import Dense

import logging 
import sys
import os 
import time 
import statistics 

logger = None 
MODEL_DIR = "models"

class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def create_critic_model(self, num_inputs, num_outputs):
        weight_initializer = keras.initializers.glorot_uniform()

        model = Sequential()

        model.add(Dense(16, activation='relu', input_shape=(num_inputs,), kernel_initializer=weight_initializer))
        model.add(Dense(16, activation='relu', kernel_initializer=weight_initializer))
        model.add(Dense(16, activation='relu', kernel_initializer=weight_initializer))
        model.add(Dense(num_outputs))

        return model

    def __init__(self, model, model_lr, model_weights, critic_model, critic_lr, num_inputs, N=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model = model
        if model_weights:
            self.model.load_weights(model_weights)

        model_opt = keras.optimizers.adam(lr=model_lr)
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=model_opt)

        if critic_model:
            self.critic_model = keras.models.load_model(critic_model)
        else:
            self.critic_model = self.create_critic_model(num_inputs, num_outputs=1)
            critic_opt = keras.optimizers.adam(lr=critic_lr)
            self.critic_model.compile(loss=keras.losses.mean_squared_error, optimizer=critic_opt)

        logger.info("Critic model:")
        self.critic_model.summary(print_fn=lambda x: logger.info(x))
        self.N = N


    def train(self, env, num_eps, batch_size=1, gamma=1.0, 
                r_mod_factor=0.01,
                report_interval=100, test_interval=500, render=False,
                model_name="default"):

        report_goals = []
        best_score = None 
        best_score_epoch = None

        for ep in range(num_eps):
            actions = []
            states = []
            goals = []
            summed_rewards = [] # intermediate rewards used to calculate goals

            critic_values = []
            for _ in range(batch_size):
                states_ep, actions_ep, rewards_ep = self.generate_episode(env, render=render)
                report_goals.append(sum(rewards_ep))
                rewards_ep = list(map(lambda x: x * r_mod_factor, rewards_ep))

                critic_values_ep = [self.critic_model.predict(s_i) for s_i in states_ep]
                
                ep_len = len(states_ep)

                reward_sum = 0 
                goals_ep = []
                summed_rewards_ep = []
                gamma_N = gamma ** self.N
                # rewards_ep = rewards_ep[::-1]
                for i in range(ep_len):
                    t = ep_len - i - 1

                    V_end = 0 if (t + self.N >= ep_len) else critic_values_ep[t + self.N]

                    reward_sum += rewards_ep[t]
                    R_t = gamma_N * V_end + reward_sum
                    goal = R_t - critic_values_ep[t]
                    goals_ep.append(goal)
                    summed_rewards_ep.append(R_t)
                    reward_sum *= gamma 

                goals += goals_ep[::-1]
                actions += actions_ep 
                states += states_ep 
                summed_rewards += summed_rewards_ep[::-1]

                assert(len(goals) == len(actions))
                assert(len(actions) == len(states))
                assert(len(states) == len(summed_rewards))

            if ep % report_interval == 0:
                logger.info(f"{'-'*10}Episode {ep}")
                logger.info(f"Average reward: {sum(report_goals) / report_interval / batch_size}")
                if ep > 0:
                    logger.info(f"Std: {statistics.stdev(report_goals)}")
                report_goals = []
            if ep % test_interval == 0:
                total_reward = self.test(env, alt_logger=logger, render=render)
                if best_score == None or total_reward > best_score:
                    logger.info(f"New best reward of {total_reward}")
                    if best_score != None: 
                        logger.info(f"Beat old reward of {best_score}")
                    best_score = total_reward 
                    best_score_epoch = ep
                self.model.save_weights(f"{MODEL_DIR}/{model_name}_{ep}.h5")
                self.critic_model.save(f"{MODEL_DIR}/{model_name}_Critic_{ep}.h5")

            
            action_vecs = np.concatenate(actions, axis=0)
            state_vecs = np.concatenate(states, axis=0)
            goal_vec = np.array(goals).reshape((-1, 1))
            target_vecs = goal_vec * action_vecs 
            rewards_vecs = np.array(summed_rewards).reshape((-1,1))

            self.model.fit(state_vecs, target_vecs, batch_size=len(states), verbose=0)
            self.critic_model.fit(state_vecs, rewards_vecs, batch_size=len(states), verbose=0)

        logger.info(f"Best score: {best_score} at epoch {best_score_epoch}")



def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--model-weights-path', dest='model_weights_path',
                        type=str, default=None, #'LunarLander-v2-weights.h5',
                        help="Path to the actor model weights file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--model-lr', dest='model_lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-model-path', type=str, default=None)
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")
    parser.add_argument('--model-name', type=str, default="default")

    parser.add_argument('--test-only', action="store_true")

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
    global logger
    # Setup logger
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    os.environ['TZ'] = 'EST+05EDT,M4.1.0,M10.5.0'
    time.tzset()    
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
    log_file_name = f'logs/{time_str}.log'

    logger = logging.getLogger("10703")
    logger.setLevel(logging.DEBUG)

    hdlr = logging.FileHandler(log_file_name)
    hdlr.setLevel(logging.INFO)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    hdlr2 = logging.StreamHandler(sys.stdout)
    hdlr2.setLevel(logging.DEBUG)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr2)
    # end setup logger

    # Parse command-line arguments.
    args = parse_arguments()
    logger.info("A2C")
    logger.info(f"Log file: {log_file_name}")
    logger.info(f"Command line args:\n{args}")

    model_config_path = args.model_config_path
    model_weights_path = args.model_weights_path 
    num_episodes = args.num_episodes
    model_lr = args.model_lr
    critic_model_path = args.critic_model_path
    critic_lr = args.critic_lr
    gamma = args.gamma 
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Plot the learning curves.

    a2c = A2C(model, model_lr, model_weights_path, critic_model_path, critic_lr, num_inputs=env.observation_space.shape[0], N=n)
    
    if args.test_only:
        logger.info("TESTING MODE")
        test_reward = a2c.test(env, alt_logger=logger, render=render)
        print(f"Testing reward: {test_reward}")
    else:
        a2c.train(env, num_episodes, gamma=gamma, model_name=args.model_name)

    logger.info(f"Log file: {log_file_name}")

if __name__ == '__main__':
    main(sys.argv)
