import gym
from gym import spaces
from improc import *
import numpy as np
import matplotlib.pyplot as plt

class ProcessedWorm(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, target):
        
        """
        Initializes the camera, light, worm starting point.
        """
        super(ProcessedWorm, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        N_DISCRETE_STATES = 12 # 12 for 30 degree increments
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=-180, high=180, shape=2, dtype=np.uint8)
        
        self.target = target
        self.templates, self.bodies = load_templates()
        self.cam, self.task = init_instruments()
        self.bgs = self.make_bgs()
        self.bg = self.bgs[0] # Set the active background to the no-light one
        
        # Arbitrary initial point initialization
        self.head, self.old_loc = [0,0],[1,1]
        self.last_loc = self.old_loc


    def step(self, action):
        """Chooses action and returns a (step_type, reward, discount, observation)"""
        img = grab_im(self.cam, self.bg)
        worms = find_worms(img, self.templates, self.bodies, ref_pts=[self.head], num_worms=1)
        
        if worms is None:
            # Returns nans if no worm is found or something went wrong
            self.task.write(0)
            self.bg = self.bgs[0]
            print('No worm')
            return dm_env.transition(reward=0., observation=(np.nan, np.nan))
        
        # Find state
        self.worm = worms[0]
        body_dir = relative_angle(self.worm['body'], self.target)
        head_body = relative_angle(self.worm['angs'][0], self.worm['body'])
        
        # Find reward
        reward = proj(self.worm['loc']-self.last_loc, [np.cos(self.target*pi/180),-np.sin(self.target*pi/180)])
        if np.isnan(reward) or np.abs(reward)>10:
            reward = 0
        
        self.last_loc = self.worm['loc']
        # return obs, reward, done (boolean), info (dict)
        return np.array([body_dir, head_body]), reward, False, {}

        
    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self.bgs = self.make_bgs()
        self.bg = self.bgs[0]
        self.head, self.old_loc = [0,0],[1,1]
        self.last_loc = self.old_loc
        
        # Takes one step and returns the observation
        return self.step(0)[0]
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        img = grab_im(self.cam, self.bg)
        worms = find_worms(img, self.templates, self.bodies, ref_pts=[self.head], num_worms=1)
        print(worms)
        plt.imshow(worms['img'])

    """ BELOW: UTILITY FUNCTIONS """
    def make_bgs(self, light_vec=[0,1], total_time=20):
        """Makes background images and stores in self"""
        self.bgs = make_vec_bg(self.cam,self.task,light_vec,total_time=total_time)
        
    def test_cam(self,bg=None):
        """To check camera is working, with or without background subtraction"""
        return grab_im(self.cam, bg)
    
    def check_ht(self):
        """
        Run this function every few seconds to make sure HT orientation is correct
        Returns True if switched.
        """
        self.head, SWITCH = ht_quick(self.worm, self.old_loc)
        self.old_loc = self.worm['loc']
        return SWITCH