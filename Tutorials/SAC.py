###################
# SAC: https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
# Discrete: https://github.com/alirezakazemipour/Discrete-SAC-PyTorch
# Both MIT license
###################


import time
import numpy as np 
from itertools import count
import itertools
import math 
import random  
from operator import itemgetter 
import os 
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.categorical import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''##########################################################################
DISCRETE SAC FUNCTIONS
'''##########################################################################

class DSAC(object):
    def __init__(self, num_inputs, num_actions, 
            gamma=.99,
            tau=.005, # target smoothing coefficient
            alpha=.2, # temperature parameter for entropy term
            target_update_interval=1, # Steps per value target update
            automatic_entropy_tuning=True, 
            hidden_size=128,
            lr=.001, # Adam optimizer
            pol_weight_decay=1e-5,
            pol_dropout=0.0, # Probability to be zeroed
            q_weight_decay=1e-5,
            q_dropout=0.0,
            ):

        # Hyperparameter definitions
        torch.autograd.set_detect_anomaly(True)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = device

        # Setting up critic/Q networks
        self.critic = DiscreteQNetwork(num_inputs, num_actions, hidden_size, dropout=q_dropout).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr, weight_decay=q_weight_decay)
        self.critic_target = DiscreteQNetwork(num_inputs, num_actions, hidden_size, dropout=q_dropout).to(self.device)

        # Automatic entropy tuning
        self.target_entropy = 0.98*(-np.log(1/num_actions))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr) 
        
        # Setting up actor/policy networks
        self.policy = DiscretePolicy(num_inputs, num_actions, hidden_size, dropout=pol_dropout).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr, weight_decay=pol_weight_decay)

    def select_action(self, state, eval=False):
        # Chooses an action with the policy network.
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action = self.policy.sample(state)
        else:
            action = self.policy.sample(state,greedy=True)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Updates parameters using a batch pulled from memory.

        # Sample a batch from memory
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Finding q-value target
        with torch.no_grad():
            _, next_probs = self.policy(next_state_batch)
            next_log_probs = torch.log(next_probs)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) 
            next_v = (next_probs*(min_qf_next_target-self.alpha*next_log_probs)).sum(-1).unsqueeze(-1)
            next_q_value = reward_batch + mask_batch * self.gamma * next_v

        qf1, qf2 = self.critic(state_batch) # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = qf1.gather(1,action_batch.unsqueeze(1).type(torch.int64)), qf2.gather(1,action_batch.unsqueeze(-1).type(torch.int64))
            
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        # Finding policy target
        _, probs = self.policy(state_batch)
        log_probs = torch.log(probs)
        with torch.no_grad():
            qf1_pi, qf2_pi = self.critic(state_batch)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (probs * (self.alpha.detach() * log_probs - min_qf_pi)).sum(-1).mean()


        # Take an optimization step for actor network
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Take an optimization step for critic network
        self.critic_optim.zero_grad()
        (qf1_loss+qf2_loss).backward()
        self.critic_optim.step()

        # Automatic entropy tuning optimization
        log_probs = (probs * log_probs).sum(-1)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()


        # For moving the target network at the set interval
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()

    def finish_model(self):
        # For when training is done: sets to eval mode if applicable.
        self.policy.eval()
        self.critic.eval()

    # Save model parameters, actor and critic as separate files within models/ folder
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

    # Load averaged models as in stochastic weight averaging
    def load_averaged_models(self, model_paths):
        actor_weights = {}
        critic_paths = {}

        for model_path in model_paths:
            actor_weights[model_path] = torch.load(f'models/sac_actor_{model_path}_')


class DiscretePolicy(nn.Module):
    # Policy network definition
    def __init__(self, num_inputs, num_actions, hidden_dim, dropout=0.0):
        super(DiscretePolicy, self).__init__()

        # Two dense layers followed by optional dropout
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.do = nn.Dropout(dropout)

        # Make probability
        self.logits = nn.Linear(hidden_dim,num_actions)
        self.apply(weights_init_)

    def forward(self, state):
        # Returns probabilities of actions given state input
        # Uses relu activation functions between layers
        x = F.relu(self.linear1(state))
        x = self.do(x)
        x = F.relu(self.linear2(x))
        logits = self.logits(x)
        probs = F.softmax(logits,-1)
        z = probs==0.0
        z = z.float()*1e-8
        return Categorical(probs), probs+z

    def sample(self, state, greedy=False):
        # Sample from probability distribution if not greedy; otherwise do max
        self.eval()
        with torch.no_grad():
            dist, p = self.forward(state)
            if greedy:
                action = p.argmax(-1)
            else:
                action = dist.sample()
        self.train()
        return action

    def to(self, device):
        return super(DiscretePolicy, self).to(device)
    
class DiscreteQNetwork(nn.Module):
    # Double Q-network definition
    def __init__(self, num_inputs, num_actions, hidden_dim, dropout=0.0):
        super(DiscreteQNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.do = nn.Dropout(dropout)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        # Send state through both networks and return estimated Q-values
        x1 = F.relu(self.linear1(state))
        x1 = self.do(x1)
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(state))
        x2 = self.do(x2)
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2


'''##########################################################################
From sac/replay_memory.py
MEMORY FUNCTIONS
'''##########################################################################

class ReplayMemory:
    # Replay buffer structure
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # Push a collected memory to the buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch):
        # Push a batch to the buffer
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            self.position = len(batch) - len(self.buffer) + self.position

    def sample(self, batch_size):
        # Sample a batch randomly from maintained buffer
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_all_batch(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)

'''
From sac/utils.py
'''
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    # Weight initialization function, Xavier
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


'''
Ensemble functions
'''

class Ensemble(nn.Module):
    # Initialize list of actors
    def __init__(self, actors, actions=2):
        super(Ensemble, self).__init__()
        self.actors = nn.ModuleList(actors).to(device)
        self.actions = actions

    def forward(self, state):
        # Return action by averaging over ensemble choices
        sums = torch.cat([act(state)[1].detach().view(state.shape[0],self.actions,1) for act in self.actors], dim=2)
        sums = torch.mean(sums, dim=2)
        return torch.argmax(sums, dim=1)

    def sample(self, state, greedy=False):
        # Sample from ensemble: if not greedy, choose from one actor
        if not greedy:
            return random.choice([act.sample(state) for act in self.actors])
        else:
            return self.forward(state)
    
    
def load_actors(prelabel, postlabel, seeds,
               frames=15, actions=2, nrns=64):
    # Loads list of actor networks
    actors = [DiscretePolicy(frames*6, actions, nrns, dropout=0.0) for i in range(seeds)]
    for i in range(seeds):
        actor_path = f'{prelabel}_s{i}_{postlabel}'
        actors[i].load_state_dict(torch.load(actor_path))
    return actors