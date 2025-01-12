import gymnasium
import flappy_bird_gymnasium
import torch
from torch import nn
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
import random
import itertools
import argparse

from NeuralNetwork import NeuralNetwork


REPLAY_MEMORY_SIZE = 100000
BATCH_SIZE = 64
NR_HIDDEN_NEURONS = 512  # try 256
EPSILON_INIT = 1
DECAY_RATE = 0.999995
MINIM_EPSILON_THRESHOLD = 0.2
MIN_STEPS_FOR_SYNC = 10
LEARNING_RATE_NN = 0.001  # try with 0.0001
DISCOUNT_FACTOR_GAMMA = 0.99  # try 0.99
REWARD_TO_STOP = 100000

DATE_FORMAT = "%m-%d %H:%M:%S"
STATS_DIR = "stats"
# os.makedirs(STATS_DIR, exist_ok=True)
# generate plots as images and save them on files instead of rendering them on screen
matplotlib.use('Agg')
# log files across episodes
LOG_FILE = os.path.join(STATS_DIR, 'dataFlappy.log')
# we save her the obtained model
MODEL_FILE = os.path.join(STATS_DIR, 'dataFlappy.pt')
# graph obtained across episodes
GRAPH_FILE = os.path.join(STATS_DIR, 'dataFlappy.png')

# how to get elements from buffer
# random.sample(replay_buffer, sample_size)


def optimize(batch, policy_net, target_net, optimizer, loss_function, DDQN=False):
    # making lists of all elements separately
    states, actions, new_states, rewards, terminations = zip(*batch)

    # transition from array of tuples to a 2-dim array, first dimension is batch size
    states = torch.stack(states)
    actions = torch.stack(actions)
    new_states = torch.stack(new_states)
    rewards = torch.stack(rewards)
    # from boolean to number to help with computation
    terminations = torch.tensor(terminations).float()

    with torch.no_grad():
        # DDQN is just instead of applying target net on net states
        # we apply it on the result of our policy network ( we use 2 Q tables)
        if DDQN:
            best_policy_action = policy_net(new_states).argmax(dim=1)
            target_q_value = rewards + (1 - terminations) * DISCOUNT_FACTOR_GAMMA *\
                             target_net(new_states).gather(dim=1, index=best_policy_action.unsqueeze(dim=1)).squeeze()
        else:
            target_q_value = rewards + (1 - terminations) * DISCOUNT_FACTOR_GAMMA * target_net(new_states).max(dim=1)[0]

    current_q_value = policy_net(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

    # compute the loss (target is expected, current_q is real val)
    loss = loss_function(current_q_value, target_q_value)

    # compute new gradients and update parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save_graph(all_rewards, epsilon_history):
    fig = plt.figure(1)

    mean_rewards = np.zeros(len(all_rewards))
    for x in range(len(mean_rewards)):
        # we do an average mean of only 100 elements; if we are at episode 1000 then the mean is from 900 to 1000
        mean_rewards[x] = np.mean(all_rewards[max(0, x-99):(x+1)])
    # plot on 1 row x 2 col grid, at cell 1
    plt.subplot(121)
    plt.ylabel('Mean rewards')
    plt.plot(mean_rewards, color='orange')

    # plot epsilon decays according to episodes, second element of the grid
    plt.subplot(122)
    plt.ylabel('Epsilon Decay')
    plt.plot(epsilon_history, color='orange')

    plt.subplots_adjust(wspace=1.0, hspace=1.0)

    fig.savefig(GRAPH_FILE)
    plt.close(fig)


def run_alg(training=True):
    # initializing the parameters used for stats when training
    if training:
        # initial time
        start_time = datetime.now()
        last_graph_update_time = start_time
        nr_tens_last_episode_print = 0

        # log messages
        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training started..."
        print(log_message)
        with open(LOG_FILE, "w") as f:
            f.write(log_message + '\n')

    # we use use_lidar = False, because we want to use the second option, with 12 parameters only
    # render_mode -> if is set to human it shows the game on screen, else set to None
    env = gymnasium.make("FlappyBird-v0", render_mode=None if training else "human", use_lidar=False)
    # env = gymnasium.make("CartPole-v1", render_mode=None if training else "human")

    # input data number
    nr_of_params = env.observation_space.shape[0]
    # output data number
    nr_of_actions = env.action_space.n

    # for stats after each episode
    rewards_per_episode = []
    epsilon_history = []

    # the found path/states/actions -> what we will use after training
    # first layer of hidden neurons number
    policy_net = NeuralNetwork(nr_of_params, nr_of_actions, NR_HIDDEN_NEURONS, simple=False)

    if training:
        # we hold a replay buffer, so that the images that we train the network on aren't too similar,
        # because next state images differs by a little only
        # this way we minimize chance of over-fitting, is the same as shuffling
        replay_buffer = deque([], maxlen=REPLAY_MEMORY_SIZE)

        # we change the epsilon value across episodes
        # used in epsilon greedy algorithm for picking our action
        epsilon = EPSILON_INIT

        # we give the same weights and biases to target net as policy net, crated for training
        # this network is basically going to be used  as aim
        target_net = NeuralNetwork(nr_of_params, nr_of_actions, NR_HIDDEN_NEURONS, simple=False)
        target_net.load_state_dict(policy_net.state_dict())

        # optimizer for the policy network
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE_NN)
        loss_function = nn.MSELoss()

        # after a number of steps we update target net
        step_count = 0

        # track the best reward
        best_reward = -99999
    else:
        # load policy net to dictated moves
        policy_net.load_state_dict(torch.load(MODEL_FILE))

        # switch to eval mode
        policy_net.eval()
        test_instance = 1

    # itertools, because we are going to stop the algorithm ourselves
    for episode in itertools.count():
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float)
        episode_reward = 0.0
        # while the agent is alive
        while episode_reward < REWARD_TO_STOP:
            # here we use the epsilon greedy algorithm:
            # if a random number between 0 and 1 is smaller than epsilon then we pick a random solution
            # else we take the best solution. Epsilon decays in time
            if training and random.random() < epsilon:
                action = env.action_space.sample()
                action = torch.tensor(action, dtype=torch.int64)
            else:
                # we stop computation of the gradients, because we are just taking beast found action,
                # and not training it now -> saving processing power
                with torch.no_grad():
                    # in pytorch the first dimension is the batch dimension
                    # now state is just a 1 dimensional tensor
                    # we add another dimension in the beginning by unsqueeze function
                    action = policy_net(state.unsqueeze(dim=0)).squeeze().argmax()

            # Take the step -> remember, action is a tensor, we take the real value by calling item on it:
            new_state, reward, terminated, _, info = env.step(action.item())
            episode_reward += reward
            # convert to tensors(for pytorch)
            new_state = torch.tensor(new_state, dtype=torch.float)
            reward = torch.tensor(reward, dtype=torch.float)

            # if training append in replay_buffer
            if training:
                replay_buffer.append((state, action, new_state, reward, terminated))
                # increment the counter for updating target network
                step_count += 1

            # move to the next state
            state = new_state

            # Checking if the player is still alive
            if terminated:
                break

        rewards_per_episode.append(episode_reward)

        # add some stats and change parameters at the end of each episode if training
        if training:
            # update model and logs if found a better rewar
            if episode_reward > best_reward:
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} (+{(episode_reward - best_reward)/best_reward * 100:0.01f}%) at episode {episode}, saving model..."
                print(log_message)
                with open(LOG_FILE, 'a') as f:
                    f.write(log_message + '\n')

                torch.save(policy_net.state_dict(), MODEL_FILE)
                best_reward = episode_reward

            # we are updating the progress graph every few seconds
            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=10):
                save_graph(rewards_per_episode, epsilon_history)
                last_graph_update_time = current_time
                if nr_tens_last_episode_print > 9:
                    print(f"At episode: {episode}")
                    nr_tens_last_episode_print = 0
                else:
                    nr_tens_last_episode_print += 1

            # decrease epsilon after an episode (for epsilon greedy-alg):
            epsilon = max(MINIM_EPSILON_THRESHOLD, epsilon * DECAY_RATE)
            epsilon_history.append(epsilon)

            # if we collected enough states we apply Q learning
            if len(replay_buffer) > BATCH_SIZE:
                batch = random.sample(replay_buffer, BATCH_SIZE)
                optimize(batch, policy_net, target_net, optimizer, loss_function, DDQN=False)

                # update the target network, only after the number of steps/actions has been achieved:
                # why we use 2 nets and we don't make this update every time after an action?
                # because the target values would continuously change and cant provide stability
                # why after a certain number of steps and not at the end of each iteration? because episodes
                # might vary in length and therefore there would be a lot of inconsistency
                if step_count > MIN_STEPS_FOR_SYNC:
                    # update means sync with policy net
                    target_net.load_state_dict(policy_net.state_dict())
                    step_count = 0
        else:
            print(f"Episode reward: {episode_reward} at test instance: {test_instance}")
            test_instance += 1

        # env.close()


# run_alg(training=True)
run_alg(training=False)

# 30min -> 13



