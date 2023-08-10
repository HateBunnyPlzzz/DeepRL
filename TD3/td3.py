# Clipped Double-Q Learning, 
# smaller one forms targets in bellman error loss

# Delayed Policy updates, policy update slower than Q-updates

# Target Policy Smoothing, adds noise to target action so policy network
# cant exploit Q approximation errors

import torch 
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions) -> None:
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, states_, actions, rewards, dones

#tells the agent, the value of a particular state.
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                name, chkpt_dir='tmp/td3'):

        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_action = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc2_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)   

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(torch.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

# updates the policy -> determine what action to take
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                n_actions, name, chkpt_dir='tmp/td3'):
        
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = torch.tanh(self.mu(prob))

        return mu

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
    update_actor_interval=2, warmup=1000, n_actions=2, max_size=1000000,
    layer1_size=400, layer2_size=300, batch_size=100, noise=0.1) -> None:
        self.gamma = gamma
        self.tau = tau # param to tell how to perform soft-update networks
        self.max_action = env.action_space.high #track of max-min actions
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
        layer2_size, n_actions=n_actions, name='actor')

        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
        layer2_size, n_actions=n_actions, name='critic_1')
        
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
        layer2_size, n_actions=n_actions, name='critic_2')

        #target nets, no backprop needed, copied parameters
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
        layer2_size, n_actions=n_actions, name='target_actor')
        
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
        layer2_size, n_actions=n_actions, name='target_critic_1')

        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
        layer2_size, n_actions=n_actions, name='target_critic_2')

        self.noise = noise
        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = torch.tensor(np.random.normal(scale=self.noise, size=(self.n_actions)))
        else:
            #deterministic action
            state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + torch.tensor(np.random.normal(scale=self.noise),
        dtype=torch.float).to(self.actor.device)

        #avoiding out of scope action values
        mu_prime = torch.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        #sample out memory
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.critic_1.device)
        done = torch.tensor(done).to(self.critic_1.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.critic_1.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic_1.device)

        #calculate our update rule
        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + \
            torch.clamp(torch.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = torch.clamp(target_actions, self.min_action[0],
                                    self.max_action[0])
        
        #action-value functions for the target actions
        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)
        
        #same for the online critic network
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        #dimensional manipulation
        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = torch.min(q1_, q2_)
        
        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        #loss functions
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return
        
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -torch.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + \
                (1-tau)*target_critic_1[name].clone()
            
        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + \
                (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                (1-tau)*target_actor[name].clone()
        
        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()

