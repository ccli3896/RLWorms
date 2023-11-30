'''
Main script for training agents for ensemble.
For SLURM scripts.
'''

import numpy as np
from copy import deepcopy 
import sys
import pickle
import random

import torch

import SAC as sac
from utils_tutorial import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TranslateRotate(object):
    '''
    Translates the animal by a random amount between [-bound,+bound].
    Also rotates the animal by a random amount.
    '''
    def __init__(self, bound, frames, target_size=3, best_reward=1.5, reward_scale=2):
        self.bound = bound
        self.frames = frames
        self.target_size = target_size
        self.best_reward = best_reward
        self.reward_scale = reward_scale

    def __call__(self, sample):

        dist = lambda x,y: np.sqrt(np.sum(np.square([x,y]),axis=0))

        state, action, reward, next_state, done = deepcopy(sample)
        batch, _ = state.shape

        shifts = np.random.uniform(size=(batch,2))
        ang_shifts = np.repeat(np.random.uniform(size=(batch,1))*2*np.pi-np.pi, self.frames, axis=1)
        
        # Rotations
        # First, angles.
        st_body, st_head = self.get_body_head_angs(state) + ang_shifts # Shape is (batch, frames)
        next_st_body, next_st_head = self.get_body_head_angs(next_state) + ang_shifts # Shape (batch, frames)
        state = self.replace_angles(state, st_body, st_head)
        next_state = self.replace_angles(next_state, next_st_body, next_st_head)
        
        # Next, locations. Rotate x,y by the ang_shift.
        state = self.rotate_coords(state, ang_shifts)
        next_state = self.rotate_coords(next_state, ang_shifts)
        
        
        # Translations
        x_shift, y_shift = shifts[:,0], shifts[:,1]
        x_shift, y_shift = (x_shift-.5)*2*self.bound, (y_shift-.5)*2*self.bound

        state[:,-2*self.frames:-self.frames] = (state[:,-2*self.frames:-self.frames]+x_shift.reshape(-1,1))/self.bound
        next_state[:,-2*self.frames:-self.frames] = (next_state[:,-2*self.frames:-self.frames]+x_shift.reshape(-1,1))/self.bound
        state[:,-self.frames:] = (state[:,-self.frames:]+y_shift.reshape(-1,1))/self.bound
        next_state[:,-self.frames:] = (next_state[:,-self.frames:]+y_shift.reshape(-1,1))/self.bound
        

        # Reward definitions
        next_dists = dist(next_state[:,-1],next_state[:,-self.frames-1])
        default_reward = -( next_dists - dist(state[:,-1],state[:,-self.frames-1])) 
        reward = np.where((next_dists*self.bound>self.target_size), default_reward, self.best_reward)

        return state, action, self.reward_scale*reward, next_state, done
    
    def get_body_head_angs(self, state):
        st_body_angs = np.arctan2(state[:,:self.frames], state[:,self.frames:2*self.frames])
        st_head_angs = np.arctan2(state[:,2*self.frames:3*self.frames], state[:,3*self.frames:4*self.frames])
        return st_body_angs, st_head_angs
    
    def replace_angles(self, state, st_body, st_head):
        state[:,:self.frames], state[:,self.frames:2*self.frames] = np.sin(st_body), np.cos(st_body)
        state[:,2*self.frames:3*self.frames], state[:,3*self.frames:4*self.frames] = np.sin(st_head), np.cos(st_head)
        return state
    
    def rotate_coords(self, state, ang_shifts):
        x = state[:,-2*self.frames:-self.frames].copy()
        y = state[:,-self.frames:].copy()
        state[:,-2*self.frames:-self.frames] = np.cos(ang_shifts)*x - np.sin(ang_shifts)*y
        state[:,-self.frames:] = np.sin(ang_shifts)*x + np.cos(ang_shifts)*y
        return state



def main():
	# Runs one seed of the agent.
    seed = int(sys.argv[1])
    sample_frac = 1. # How much of the dataset (float) to randomly sample.
    rseed = 0.
    line_number = int(sys.argv[2])
    print(seed,sample_frac,rseed)

    timesteps=15
    translated = 900
    batch_size = 64
    nrns = 64
    epochs = 20

    random.seed(rseed+73) 

    # First import the dataset
    with open(f'./Training data/L{line_number}.pkl','rb') as f:
        memories = pickle.load(f)
    env_pool = sac.ReplayMemory(int(1e6))
    #memories = random.sample(memories, int(len(memories)*sample_frac))
    for mem in memories:
        env_pool.push(*mem,False)

    # Setting translation function and HPs

    tr = TranslateRotate(translated, timesteps, target_size=30, best_reward = 2., reward_scale=2.)
    training = True

    # Initializing agent
    agent_name = f'L{line_number}_{rseed}sample{int(sample_frac*100)}_s{seed}'
    agent = sac.DSAC(timesteps*6, 2, hidden_size=nrns, gamma=0.99, 
            lr=0.001, 
            target_update_interval=500,
            pol_dropout=0., pol_weight_decay=0.,
            q_dropout=0., q_weight_decay=0.,)
    agent.save_model(f'{agent_name}_init')

    # Training and saving loop
    for epoch in np.arange(epochs):
        for total_step in range(5000):
            if total_step%1000==0:
                print('Step', total_step)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = tr(env_pool.sample(batch_size))
            batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
            batch_done = (~batch_done).astype(int)
            if training is True:
                agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), batch_size, total_step)

            if training is True and total_step%5000==0:
                agent.save_model(f'{agent_name}_e{epoch}')
    agent.finish_model()
    agent.save_model(f'{agent_name}_fin')




if __name__=='__main__':
  main()
