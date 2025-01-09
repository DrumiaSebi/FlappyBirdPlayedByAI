import flappy_bird_gymnasium
from torch import nn
import gymnasium
from collections import deque
import random
import itertools

import torch

from NeuralNetwork import NeuralNetwork


REPLAY_MEMORY_SIZE = 100000
BATCH_SIZE = 32
NR_HIDDEN_NEURONS = 256
EPSILON_INIT = 1
DECAY_RATE = 0.9995
MINIM_EPSILON_THRESHOLD = 0.05
MIN_STEPS_FOR_SYNC = 1000
LEARNING_RATE_A = 0.005
DISCOUNT_FACTOR_GAMMA = 0.99

# how to get elements from buffer
# random.sample(replay_buffer, sample_size)


def optimize(batch, policy_net, target_net, optimizer, loss_function):
    # making lists of all elements separately
    states, actions, new_states, rewards, terminations = zip(*batch)

    # transition from array of tuples to a 2-dim array
    states = torch.stack(states)
    actions = torch.stack(actions)
    new_states = torch.stack(new_states)
    rewards = torch.stack(rewards)
    # from boolean to number to help with computation
    terminations = torch.tensor(terminations).float()

    with torch.no_grad():
        target_q_value = rewards + (1-terminations) * DISCOUNT_FACTOR_GAMMA * target_net(new_states).max(dim=1)[0]

    current_q_value = policy_net(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
    for state, action, new_state, reward, terminated in batch:
        # this is the new Q learning value when using neural networks -> course page 39
        if terminated:
            target_q_value = reward
        else:
            with torch.no_grad():
                target_q_value = reward + DISCOUNT_FACTOR_GAMMA * target_net(new_state).max()

        current_q_value = policy_net(state)

        # compute the loss (target is expected, current_q is real val)
        loss = loss_function(current_q_value, target_q_value)

        # compute new gradients and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def run_alg(training=True):
    # we use use_lider = False, because we want to use the second option, with 12 parameters only
    # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    env = gymnasium.make("CartPole-v1", render_mode="human")

    # input data, output data, and first layer of hidden neurons
    nr_of_params = env.observation_space.shape[0]
    nr_of_actions = env.action_space.n

    # for stats after each epoch
    rewards_per_episode = []
    epsilon_history = []

    # the found path/states/actions
    policy_net = NeuralNetwork(nr_of_params, nr_of_actions, NR_HIDDEN_NEURONS)

    # we hold a replay buffer, so that the images that we train the network on aren't too similar, because next state
    # image differs by a little only
    # this way we minimize chance of over-fitting, is the same as shuffling
    if training:
        replay_buffer = deque([], maxlen=REPLAY_MEMORY_SIZE)
        epsilon = EPSILON_INIT
        # we give the same weights and biases to target net as policy net
        target_net = NeuralNetwork(nr_of_params, nr_of_actions, NR_HIDDEN_NEURONS)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE_A)
        loss_function = nn.MSELoss()
        # after a number of steps we update target net
        step_count = 0

    # itertools, because we are going to stop the algorithm ourselves
    for episode in itertools.count():
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float)
        episode_reward = 0.0
        # while the agent is alive
        while True:
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
                # increment the counter
                step_count += 1

            # move to the next state
            state = new_state

            # Checking if the player is still alive
            if terminated:
                break

        rewards_per_episode.append(episode_reward)

        # decrease epsilon after a epoch:
        epsilon = max(MINIM_EPSILON_THRESHOLD, epsilon * DECAY_RATE)
        epsilon_history.append(epsilon)

        if len(replay_buffer) > BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            optimize(batch, policy_net, target_net, optimizer, loss_function)

            # update the target network, only after the number of steps/actions has been achieved:
            # why we use 2 nets and we dont make this update every time after an action?
            # because the target values would continuously change and cant provide stability
            # why after a certain number of steps and not at the end of each iteration? because episodes
            # might vary in length and therefore there would be a lot of inconsistency
            if step_count > MIN_STEPS_FOR_SYNC:
                # update means sync with policy net
                target_net.load_state_dict(policy_net.state_dict())
                step_count = 0

    env.close()


run_alg()



