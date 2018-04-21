#!/usr/bin/env python3
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
import time
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Lambda 
from random import sample
import logging
import time
import os.path
from replay.memory import Replay_Memory

logger = None 

models_dir = "models"
curr_model_dir = None  
logs_dir = "logs"
model_file_ext = ".h5"

EPS = 1e-6

# Performs the combining operation as specified by the dueling paper
def dueling_combine(sv_adv_layer):
    import keras 

    state_value = sv_adv_layer[0]
    advantage = sv_adv_layer[1]

    num_actions = advantage.shape[1].value 

    adv_mean = keras.backend.mean(advantage, axis=1, keepdims=True)
    
    return state_value + (advantage - adv_mean)

class QNetwork():

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    def __init__(self, environment_name, num_inputs, num_outputs, lr, deep=False):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        # weight_init = keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)

        model = Sequential()


        if deep == "deep":
            #weight_initializer = keras.initializers.RandomUniform(-0.05, 0.05)
            weight_initializer = keras.initializers.glorot_uniform()
            #weight_initializer = keras.initializers.he_normal()

            logger.info("Using deep architecture.")
            model.add(Dense(40, activation='relu', input_shape=(num_inputs,), kernel_initializer=weight_initializer))
            model.add(Dense(40, activation='relu', kernel_initializer=weight_initializer))
            model.add(Dense(30, activation='relu', kernel_initializer=weight_initializer))
            model.add(Dense(num_outputs))


        elif deep == "dueling":

            logger.info("Using dueling architecture.")

            weight_initializer = keras.initializers.glorot_uniform()
            weight_initializer_dense = keras.initializers.RandomUniform(-0.05, 0.05)

            inputs = Input(shape=(num_inputs, ))
            d = Dense(25, activation='relu', kernel_initializer=weight_initializer)(inputs)
            d = Dense(25, activation='relu', kernel_initializer=weight_initializer)(d)
            
            # Stream 1
            state_value = Dense(10, activation='relu', kernel_initializer=weight_initializer)(d)
            state_value = Dense(10, activation='relu', kernel_initializer=weight_initializer)(state_value)
            state_value = Dense(1, activation='linear', name="state_value")(state_value)

            # Stream 2
            advantage = Dense(10, activation='relu', kernel_initializer=weight_initializer)(d)
            advantage = Dense(10, activation='relu', kernel_initializer=weight_initializer)(advantage)
            advantage = Dense(num_outputs, activation='linear', name="advantage")(advantage)

            # Combine
            q_values = Lambda(dueling_combine, output_shape=(num_outputs,))([state_value, advantage])

            model = Model(inputs=inputs, outputs=q_values)


        else:
            logger.info("Using linear architecture.")

            weight_initializer = keras.initializers.RandomUniform(-0.05, 0.05)
            weight_initializer = keras.initializers.RandomUniform(-1, 0)
            model.add(Dense(num_outputs, input_shape=(num_inputs,), kernel_initializer=weight_initializer))

        logger.info(f"Using initial learning rate {lr}")
        opt = keras.optimizers.adam(lr=lr)
        #opt = keras.optimizers.SGD(lr=lr)

        model.compile(optimizer=opt, loss="mean_squared_error", metrics=["accuracy"])

        target_model = keras.models.clone_model(model)
        target_model.set_weights(model.get_weights())

        self.model = model 
        self.target_model = target_model 

        self.num_obs = num_inputs
        self.num_actions = num_outputs
        
    def predict(self, x):
        assert(x.ndim == 2)

        batch_size = x.shape[0]
        return self.model.predict(x, batch_size), self.target_model.predict(x, batch_size)

    def train(self, x, y, batch_size=1):
        assert(x.ndim == 2)

        assert(x.shape[0] == y.shape[0])
        assert(y.shape[1] == self.num_actions)

        # self.model.train_on_batch(x, y)
        self.model.fit(x, y, batch_size=batch_size, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, model_name):
        self.model.save(curr_model_dir + model_name + model_file_ext)

    def load_model(self, model_name):
        # Helper function to load an existing model.
        self.model = keras.models.load_model(curr_model_dir + model_name + model_file_ext, custom_objects={"keras": keras})


class DQN_Agent():

    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    
    def __init__(self, c_model_dir, c_logger, environment_name, gamma, lr_init, eps_init=0.5, test_mode=False, 
        model_name=None, render=False, deep=False, seed=None, alt_learn=False, 
        goal_size=0, combined_replay=False, priority_replay=False, hindsight_replay=False):


        global curr_model_dir
        global logger 

        curr_model_dir = c_model_dir 
        logger = c_logger

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 
        self.env_name = environment_name
        self.env = gym.make(environment_name)

        if seed != None:
            logger.info(f"Gym seed set to {seed}")
            self.env.seed(seed)
        self.seed = seed

        num_obs = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.net = QNetwork(environment_name, num_obs + goal_size, num_actions, lr_init, deep=deep)
        if test_mode:# or os.path.exists(model_name + model_file_ext):
            assert(model_name)
            self.net.load_model(model_name)
        
        self.gamma = gamma
        self.eps_init = eps_init
        self.eps = eps_init 
        self.alt_learn = alt_learn

        self.model_name = model_name
        self.render = render 

        self.combined_replay = combined_replay
        self.priority_replay = priority_replay
        self.hindsight_replay = hindsight_replay

        if combined_replay:
            logger.info("Using combined replay")
        if priority_replay: 
            logger.info("Using prioritized replay") 
        if hindsight_replay:
            logger.info("Using hindsight replay") 



    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.         
        rand_prob = np.random.rand(1)
        if rand_prob < self.eps:
            return np.random.randint(len(q_values))
        else:
            return np.argmax(q_values)

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time. 
        return np.argmax(q_values)

    def take_step(self, curr_state, batch_size, default_goal=None):
        curr_state = curr_state.reshape((1, -1))
        q_values_pred, q_values_target = self.net.predict(curr_state)
        action_i = self.epsilon_greedy_policy(q_values_pred)
        next_state, reward, is_terminal, debug_info = self.env.step(action_i)
        next_state = next_state.reshape((1, -1))
        if default_goal != None:
            next_state = np.concatenate((next_state, default_goal), axis=1)

        return (curr_state, reward, action_i, next_state, is_terminal) 


    def calc_target(self, experience, q_values=None):
        
        (curr_state, reward, action_i, next_state, is_terminal) = experience

        if q_values == None:
            q_values_pred, q_values_target = self.net.predict(curr_state)
        
        next_q_values_pred, next_q_values_target = self.net.predict(next_state)
        predictions_target = np.amax(next_q_values_target, axis=1)
        #logger.info(f"reward: {reward.shape}, is_terminal: {is_terminal.shape}, predcitions: {predictions.shape}")

        y = np.zeros(shape=action_i.shape)

        for idx in range(action_i.shape[0]):
            if is_terminal[idx] or (self.alt_learn and next_state[idx, 0] > 0.5):
                y[idx] = reward[idx]
            else:
                y[idx] = reward[idx] + self.gamma * predictions_target[idx]

        #logger.info(f"y: {y.shape}")

        target = q_values_pred.copy()
        #logger.info(f"target: {target.shape}")

        #logger.info(f"{action_i} target: {target} indexed: {target[:, action_i]}")
        
        for idx in range(action_i.shape[0]):
            target[idx, action_i[idx]] = y[idx]

        
        pred_max_action = np.zeros(shape=action_i.shape) 
        for idx in range(q_values_pred.shape[0]):
            pred_max_action[idx] = q_values_pred[idx, action_i[idx]]
        assert(y.shape == pred_max_action.shape)

        td_errors = y - pred_max_action 
        
        return target, td_errors


    def train(self, use_episodes, episodes_limit, steps_limit, rep_batch_size=False, save_best=True, default_goal=None):
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 
        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.

        if self.hindsight_replay:
            assert(default_goal)

        batch_size = 1
        print_episode_mod = 200 # print every
        test_episode_mod = 200  

        save_episode_mod = 300
        save_steps_mod = 100000

        update_target_model_eps_mod = 20 # 50

        priority_replay_alpha = 0.6
        priority_replay_beta_init = 0.4
        priority_replay_beta = priority_replay_beta_init # TODO: anneal
        

        num_episodes = 0
        num_total_steps = 0

        train_average = 0 
        train_variance = []

        rep_mem = Replay_Memory(prioritized=self.priority_replay, hindsight=self.hindsight_replay, default_goal=default_goal, priority_alpha=priority_replay_alpha)

        if rep_batch_size:
            self.burn_in_memory(rep_mem, default_goal)
       
        counter = num_episodes if use_episodes else num_total_steps
        counter_limit = episodes_limit if use_episodes else steps_limit
        save_mod = save_episode_mod if use_episodes else save_steps_mod
        save_steps_ct = -1
        save_video_episode_mod = counter_limit // 3 # Want videos every 1/3 of training process
        save_video_steps_ct = -1
        

        logger.info("Training for {} {}".format(counter_limit, "episodes" if use_episodes else "steps"))

        best_test = None

        # while num_episodes < episodes_limit:
        # while num_total_steps < steps_limit:
        while counter < counter_limit:

            if num_episodes % print_episode_mod == 0:
                logger.info(f"Episode {num_episodes}")
                logger.info(f"Step {num_total_steps}")

            initial_state = self.env.reset().reshape((1, -1))
            if self.hindsight_replay:
                initial_state = np.concatenate((initial_state, default_goal), axis=1)
            curr_state = initial_state

            num_ep_steps = 0 
            ep_reward = 0

            # Run till episode termination
            is_terminal = False

            # Go through an episode
            episode_exps = []
            while not is_terminal:
                # self.eps = max(self.eps_init / 5, self.eps_init * (1 - 0.8 / 100000 * ((num_ep_steps + num_total_steps))))
                self.eps = max(0.05, self.eps_init * (1 - 0.8 / 100000 * ((num_ep_steps + num_total_steps))))

                exp_batch = []

                for i in range(batch_size):
                    # exp_batch = [] # added line

                    experience = self.take_step(curr_state, batch_size, default_goal)
                    _, reward, action_i, curr_state, is_terminal = experience
                    exp_batch.append((experience[0].copy(), reward, action_i, curr_state.copy(), is_terminal))
                    
                    if is_terminal:
                        break
               
                if reward == 0:
                    logger.info("NOTICE THIS: Yay reached the top " + "*"*100);
                
                episode_exps.extend(exp_batch)

                ep_reward += reward
                num_ep_steps += 1
            
                # if is_terminal:
                #     break
                
                train_batch = exp_batch
                train_size = len(exp_batch)

                if rep_batch_size:
                     
                    # lock rep_batch_size
                    if self.combined_replay:
                        rep_batch_size -= 1 
                    else:
                        # Add current transition to buffer first if not using combined replay
                        placed_index = rep_mem.append(exp_batch[0])

                    # TODO: incorporate weights into loss function calculation
                    train_batch, batch_weights, batch_indexes = rep_mem.sample_batch(rep_batch_size, beta=priority_replay_beta)

                    if self.combined_replay:
                        assert(len(exp_batch) == 1)
                        placed_index = rep_mem.append(exp_batch[0])
                        train_batch.append(exp_batch[0])

                        if self.priority_replay:
                            batch_indexes.append(placed_index)

                        rep_batch_size += 1

                    # unlock rep_batch_size

                    train_size = rep_batch_size

                #TODO implement goal augmentation for non memory replay.
                    
                exp_arr_list = [np.reshape(np.array([exp[i] for exp in train_batch]), (train_size, -1)) for i in range(5)]
                
                curr_states = exp_arr_list[0]
                targets, td_errors = self.calc_target(tuple(exp_arr_list))
                self.net.train(curr_states, targets, batch_size=train_size)

                if self.priority_replay:
                    rep_mem.update_priorities(batch_indexes, np.abs(td_errors) + EPS)
                    
                if (not use_episodes and (num_total_steps + num_ep_steps) > steps_limit):
                    break
            if self.hindsight_replay:
                rep_mem.append_episode(episode_exps)


            #logging/saving/recording section
            if self.env_name == "MountainCar-v0" and ep_reward > -200:
                logger.info(f"*****Got better reward: {ep_reward} on ep {num_episodes}")

            train_average += ep_reward / print_episode_mod
            train_variance += [ep_reward]
            if num_episodes % print_episode_mod == 0:
                logger.info("Episode reward mean: {:.3f}".format(train_average))
                logger.info("Episode reward std: {:.3f}".format(np.sqrt(np.sum(np.square(np.array(train_variance) - train_average)) / print_episode_mod)))
                train_average, train_variance = (0, [])
                

            if num_episodes % test_episode_mod == 0:
                _, result = self.test(num_episodes=20, hindsight=self.hindsight_replay, default_goal=default_goal)
                
                if save_best:
                    if best_test == None or best_test < result:
                        model_name_ep = self.model_name
                        self.net.save_model(model_name_ep)
                        logger.info(f"Beat previous value of {best_test}! Saved model to {curr_model_dir + model_name_ep + model_file_ext}")
                        best_test = result
            
            # if counter // save_mod > save_steps_ct:
            #     model_name_ep = f"{self.model_name}_{counter // save_mod}_of_{counter_limit // save_mod}"
            #     self.net.save_model(model_name_ep)
            #     logger.info(f"Saved model to {curr_model_dir + model_name_ep + model_file_ext}")

            #     save_steps_ct += 1

            if counter // save_video_episode_mod > save_video_steps_ct:
                model_name_ep = f"{self.model_name}_{counter // save_video_episode_mod}_of_{counter_limit // save_video_episode_mod}"
                self.net.save_model(model_name_ep)
                logger.info(f"Saved model for rendering to {curr_model_dir + model_name_ep + model_file_ext}")

                save_video_steps_ct += 1          
            #end logging/saving/recording section
                

            if num_episodes % update_target_model_eps_mod == 0:
                # logger.info(f"Updated target network")
                self.net.update_target_model()

            num_total_steps += num_ep_steps     
            num_episodes += 1

            counter = num_episodes if use_episodes else num_total_steps

        # Save 3/3 video
        model_name_ep = f"{self.model_name}_{counter // save_video_episode_mod}_of_{counter_limit // save_video_episode_mod}"
        self.net.save_model(model_name_ep)
        logger.info(f"Saved model for rendering to {curr_model_dir + model_name_ep + model_file_ext}")

        if self.seed != None:
            logger.info(f"Reminder that gym seed set to {self.seed}")

    def test(self, num_episodes=100, record_video=False, hindsight=False, default_goal=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory. 

        logger.info("**Testing mode")
        
        if record_video:
            logger.info("Recording video from loaded files and then exiting")

            # Record every episode that we run
            self.env = gym.wrappers.Monitor(self.env, "frames/" + self.model_name, force=True, video_callable=(lambda ep_i: True), mode="evaluation")

            num_episodes = 4 # 0/3, 1/3, 2/3, 3/3

        logger.info("Running for {} episodes".format(num_episodes))

        all_rewards = []
        total_reward = 0    


        self.eps = 0 #self.eps_init / 10
        for ep_i in range(num_episodes):

            if record_video:
                logger.info(f"Loading model {self.model_name}_{ep_i}_of_{3}")
                self.net.load_model(f"{self.model_name}_{ep_i}_of_{3}")

            initial_state = self.env.reset().reshape((1, -1))
            if hindsight:
                initial_state = np.concatenate((initial_state, default_goal), axis=1)
            curr_state = initial_state

            ep_reward = 0

            while True:
                curr_state = curr_state.reshape((1, -1))
                q_values_pred, q_values_target = self.net.predict(curr_state)
                action_i = self.epsilon_greedy_policy(q_values_pred)

                curr_state, reward, is_terminal, debug_info = self.env.step(action_i)

                if hindsight:
                    curr_state = curr_state.reshape((1, -1))
                    curr_state = np.concatenate((curr_state, default_goal), axis=1)
           

                if self.render:
                    self.env.render() 
                    time.sleep(0.00001)
                
                ep_reward += reward 
                if is_terminal:
                    break

            total_reward += ep_reward 
            all_rewards.append(ep_reward)

        average_total_reward = total_reward / num_episodes
        reward_stdev = np.std(np.array(all_rewards))

        logger.info(f"(test) Average reward per episode: {average_total_reward}")
        logger.info(f"Standard deviation: {reward_stdev}")

        return total_reward, average_total_reward



    def burn_in_memory(self, rep_mem, default_goal=None):
        curr_state = self.env.reset().reshape((1, -1))
        if rep_mem.hindsight:
            curr_state = np.concatenate((curr_state, default_goal), axis=1)
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        for i in range(rep_mem.burn_in):
            experience = self.take_step(curr_state, 1, default_goal)
            (prev_state, reward, action, curr_state, is_terminal) = experience
            rep_mem.append((prev_state.copy(), reward, action, curr_state.copy(), is_terminal))
            if is_terminal:
                curr_state = self.env.reset().reshape((1, -1))
                if rep_mem.hindsight:
                    curr_state = np.concatenate((curr_state, default_goal), axis=1)
    
