import numpy as np
from collections import deque
# deque is a double sided pop and insert array structure for fast i/o

import matplotlib.pyplot as plt

#pytorch 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym
import pygame
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#######################################
# environmet
env_id = "CartPole-v1"
env = gym.make(env_id)



class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
#env.reset() provides a set of random default observation from the environmet upon acting

#* When we calculate the return Gt, we see
#  that we calculate the sum of discounted rewards 
# starting at the timestep (t)
# Calulating the returns from T-1 to 0 timestep, the notion of Backward E-Traces
# compute the return at each timestep efficiently.
policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
def reinforce(n_episodes=1000, max_t=100, gamma=1, print_every=1):
    #calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # line 3 
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
    # line 4 Generating an episode/trajectory
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        #Line 7:
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob*R)

        policy_loss = torch.cat(policy_loss).sum()

        #Computing the gradient
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break
        
    return scores
scores = reinforce()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env = gym.make('CartPole-v0')

state = env.reset()
for t in range(1000):
    action, _ = policy.act(state)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break 

env.close()