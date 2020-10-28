import gym
from gym import spaces
import random
import numpy as np
import matplotlib.pyplot as plt
import time

def add_to_traj(trajectory,info):
    # appends each key in info to the corresponding key in trajectory.
    # If trajectory is empty, returns trajectory as copy of info but with each element as list
    # so it can be appended to in the future.

    if trajectory:
        for k in info.keys():
            trajectory[k].append(info[k])
    else:
        for k in info.keys():
            trajectory[k] = [info[k]]

class FakeWorm(gym.Env):
    """
    Custom Environment that follows gym interface
    Chooses a random state at each step and calculates rewards based on Gaussians centered on -1 or 1,
    depending on deterministic policy
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        
        """
        Initializes the camera, light, worm starting point.
        ep_len is in seconds; each episode will terminate afterward. 
        """
        super(FakeWorm, self).__init__()

        self.action_space = spaces.Box(low=0,high=1,shape=(1,),dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-180,-180]), high=np.array([180,180]), dtype=np.uint8)
        
        self.steps = 0
        self.state = np.zeros(2)


    def step(self, action):
        """Chooses action and returns a (step_type, reward, discount, observation)"""

        state_vec = np.arange(-180,180,30)
        self.steps += 1

        reward = np.random.normal()
        if self.state[0]*self.state[1] < 0 and action > .5:
            reward = np.random.normal(loc=1)
        elif self.state[0]*self.state[1] > 0 and action >.5:
            reward = np.random.normal(loc=-1)
        
        obs = np.array([random.choice(state_vec),random.choice(state_vec)])
        self.state = obs

        if self.steps > 100:
            finished = True
        else:
            finished = False
        # return obs, reward, done (boolean), info (dict)
        return obs, reward, finished, {'reward':reward}

        
    def reset(self,target=None):
        """Returns the first `TimeStep` of a new episode."""
        self.steps = 0
        
        # Takes one step and returns the observation
        self.state = self.step(0)[0]
        return self.step(0)[0]
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass