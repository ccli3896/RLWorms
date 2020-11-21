import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class FakeWorm(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, distribution_dict, start=None, ep_len=1e6):
        '''
        distribution_dict must have the following:
        body_on: [12,12,2] with dimensions [body angle, head angle, mu/sig]. 
        body_off: ''
        head_on: ''
        head_off: ''
        reward_on: ''
        reward_off: ''
        '''
        super(FakeWorm, self).__init__()
        # Setting environment parameters
        self.grid_width = 12        
        self.action_space = spaces.Discrete(2) # Light off or on.
        self.observation_space = spaces.Discrete(self.grid_width**2) 
        
        # Setting initial conditions
        if start is None:
            self._state = np.random.choice(self.grid_width**2)
        else:
            self._state = self.grid2obs(start)
        self.state = self.obs2grid(self._state)
            
        # Setting self parameters
        self.dist_dict = distribution_dict
        self.ep_len = ep_len
        self.steps = 0
        self.finished = False
    

    def step(self, action):
        """Chooses action and returns a (step_type, reward, discount, observation)"""
        self.steps += 1
        # If step count reaches episode length
        if self.steps >= self.ep_len:
            self.finished = True
            
        # Draws new reward and state from previous state 
        olds = self.grid2coords(self.state)
        if action==0:
            self.state[0] = self.get_sample('body_off', olds)
            self.state[1] = self.get_sample('head_off', olds)
            reward = self.get_sample('reward_off', olds)
        elif action==1:
            self.state[0] = self.get_sample('body_on', olds)
            self.state[1] = self.get_sample('head_on', olds)
            reward = self.get_sample('reward_on', olds)
        else:
            raise ValueError('Invalid action')
                       
        # Return obs, reward, done (boolean), info (dict)
        self._state = self.grid2obs(self.state)
        return self._state, reward, self.finished, {}

        
    def reset(self,start=None):
        """Returns the first `TimeStep` of a new episode."""
        self.finished = False
        self.steps = 0
        # Setting initial conditions
        if start is None:
            self._state = np.random.choice(self.grid_width**2)
        else:
            self._state = self.grid2obs(start)
        self.state = self.obs2grid(self._state)
        
        # Takes one step and returns the observation
        return self._state
    
    def render(self, mode='human', close=False):
        print('At',self.state)



    ''' Utility '''
    def get_sample(self,dkey,olds):
        # Returns a single sample from normal distribution given by statistics in dictionary[dkey], 
        # at location given by olds (index into matrix in dkey)
        mu,sig = self.dist_dict[dkey][olds[0],olds[1]]
        if 'reward' in dkey:
            return np.random.normal(mu,sig)
        else:
            return self.keep_inside(self.myround(np.random.normal(mu,sig)))
                       
    def myround(self, x, base=30):
        return base * round(x/base)

    def keep_inside(self,num):
        # Restricts values to [-180,180)
        if num<-180:
            num+=360
        elif num>=180:
            num-=360
        return num

    ''' Conversion functions '''
    # obs is from 0 to 143.
    # grid is [-180 to 150, -180 to 150]
    # coords is [0 to 11, 0 to 11]
    def obs2grid(self,obs):
        gridcoords = np.array([obs//self.grid_width, obs%self.grid_width])
        return (gridcoords*30)-180
    def grid2obs(self,gridcoords):
        gridcoords = ((np.array(gridcoords)+180)/30).astype(int)
        return self.grid_width*gridcoords[0] + gridcoords[1]
    def grid2coords(self,gridcoords):
        return (np.array(gridcoords)+180)//30
    def coords2grid(self,coords):
        return (np.array(coords)*30)-180
    ''' End of conversion functions '''