import gym
from gym import spaces
from improc import *
import numpy as np
import matplotlib.pyplot as plt
import time

class ProcessedWorm(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, target, ep_len=900):
        
        """
        Initializes the camera, light, worm starting point.
        ep_len is in seconds; each episode will terminate afterward. 
        """
        super(ProcessedWorm, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        N_DISCRETE_STATES = 12 # 12 for 30 degree increments
        N_DISCRETE_ACTIONS = 2 # on or off

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=np.array([-180,-180]), high=np.array([180,180]), dtype=np.uint8)
        
        self.target = target
        self.templates, self.bodies = load_templates()
        self.cam, self.task = init_instruments()
        self.bgs = self.make_bgs()
        self.bg = self.bgs[0] # Set the active background to the no-light one
        
        # Arbitrary initial point initialization
        self.head, self.old_loc = [0,0],[1,1]
        self.last_loc = self.old_loc

        self.timer = Timer(ep_len)


    def step(self, action, sleep_time=.1):
        """Chooses action and returns a (step_type, reward, discount, observation)"""
        # In info, returns worm info for that step 
        # {'img':_, 'loc':(x,y), 't':_}

        self.task.write(action)
        time.sleep(sleep_time)

        img = grab_im(self.cam, self.bg)
        worms = find_worms(img, self.templates, self.bodies, ref_pts=[self.head], num_worms=1)
        
        if worms is None:
            # Returns nans if no worm is found or something went wrong
            self.task.write(0)
            self.bg = self.bgs[0]
            print(f'No worm \t\t\r',end='')
            return np.array([np.nan,np.nan]), 0, False, {
                'img':None,
                'loc':np.array([np.nan,np.nan]),
                't':self.timer.t}
        
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

        self.timer.update()
        if self.timer.check():
            finished = True
            self.task.write(0)
        else:
            finished = False

        return np.array([body_dir, head_body]), reward, finished, {
            'img': self.worm['img'],
            'loc': self.worm['loc'],
            't': self.timer.t 
        }

        
    def reset(self,target=None):
        """Returns the first `TimeStep` of a new episode."""
        if target is not None:
            self.target = target
        self.bgs = self.make_bgs()
        self.bg = self.bgs[0]
        self.head, self.old_loc = [0,0],[1,1]
        self.last_loc = self.old_loc

        self.timer.reset()
        
        # Takes one step and returns the observation
        return self.step(0)[0]
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        img = grab_im(self.cam, self.bg)
        worms = find_worms(img, self.templates, self.bodies, ref_pts=[self.head], num_worms=1)
        if worms is None:
            # Returns nans if no worm is found or something went wrong
            self.task.write(0)
            self.bg = self.bgs[0]
            print('No worm')
            return
        worm = worms[0]
        print(worm)
        plt.imshow(worm['img'])

    """ BELOW: UTILITY FUNCTIONS """
    def make_bgs(self, light_vec=[0,1], total_time=20):
        """Makes background images and stores in self"""
        return make_vec_bg(self.cam,self.task,light_vec,total_time=total_time)
        
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

    def close(self):
        self.cam.exit()
        self.task.write(0)