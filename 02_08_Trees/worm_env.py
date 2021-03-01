import gym
from gym import spaces
from improc import *
import numpy as np
import time
from math import *
import utils as ut

##########################
# This is a version where observations are normalized to [-1,1] WITHIN this environment.
# Did it because the normalized box wrapper seemed buggy and wasn't normalizing
# the first observation for each epoch.
# Real-time HT fix with continuous direction checking.
##########################


class ProcessedWorm(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, target, ep_len=1200, bg_time=20,window=30):
        
        """
        Initializes the camera, light, worm starting point.
        ep_len is in seconds; each episode will terminate afterward. 
        """
        super(ProcessedWorm, self).__init__()
        # For clarity: last_loc is for calculating reward.

        # Define action and observation space
        # They must be gym.spaces objects

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]), dtype=np.uint8)

        self.target = target
        self.ep_len = ep_len
        self.templates, self.bodies = load_templates()

        self.bg_time=bg_time
        self.timer = Timer(ep_len)
        self.steps = 0
        self.finished = False
        self.no_worm_flag = True
        self.recent_locs = np.zeros((window,2))
        self.window = window


    def step(self, action, cam, task, sleep_time=0):
        """Chooses action and returns a (step_type, reward, discount, observation)"""
        # In info, returns worm info for that step 
        # {'img':_, 'loc':(x,y), 't':_}

        self.steps += 1

        task.write(action)
        time.sleep(sleep_time)

        # Get data
        img = grab_im(cam, self.bg)
        worms = find_worms(img, self.templates, self.bodies, ref_pts=self.head, num_worms=1)

        # Timer checks: episode end
        if self.timer.check():
            self.finished = True
        self.timer.update()

        if worms is not None:
            # Find current direction of motion.
            # If the current point is a jump from all others in the array, return same as if worms is None.
            jump_limit = 100
            self.recent_locs = np.append(self.recent_locs[1:], [worms[0]['loc']], axis=0)
            direction = None
            for i in range(self.window):
                if pt_dist(self.recent_locs[-1],self.recent_locs[i]) < jump_limit:
                    # Set direction
                    direction = np.arctan2(-(self.recent_locs[-1,1]-self.recent_locs[i,1]), 
                        self.recent_locs[-1,0]-self.recent_locs[i,0]) *180/pi
                    break    

        if worms is None or direction is None:
            # Returns zeros if worm isn't found or something went wrong.
            task.write(0)
            self.no_worm_flag = not self.no_worm_flag
            if self.no_worm_flag:
                # This adds some jitter to the print statement so I can tell
                # if the worm is gone or just popped out for a bit
                print(f'No worm \t\t\r',end='')
            else:
                print(f' No worm\t\t\r',end='')
            return np.zeros(2), 0, self.finished, {
                'loc': np.zeros(2),
                't': self.timer.t,
                'endpts': np.zeros((2,2))-1,
                'obs': np.zeros(2),
                'reward': 0,
                'target': self.target,
                'action': action,
                }
                
        # Find state 
        body_dir = relative_angle(worms[0]['body'], self.target)
        if abs(relative_angle(body_dir,direction-self.target)) > 90:
            body_dir = ut.wrap_correct(body_dir+180)
            head_body = relative_angle(worms[0]['angs'][1], worms[0]['body']+180)
            worms[0]['endpts'] = np.fliplr(worms[0]['endpts'])
        else:
            head_body = relative_angle(worms[0]['angs'][0], worms[0]['body'])

        obs = np.array([body_dir,head_body])/180.
        
        # Find reward and then update last_loc variable
        reward = proj(worms[0]['loc']-self.last_loc, [np.cos(self.target*pi/180),-np.sin(self.target*pi/180)])
        self.last_loc = worms[0]['loc']
        if np.isnan(reward) or np.abs(reward)>10:
            reward = 0
        
        self.head = worms[0]['endpts'][:,0] # for endpoint reference at next step
        # return obs, reward, done (boolean), info (dict)
        return obs, reward, self.finished, {
            #'img': worms[0]['img'],
            'loc': worms[0]['loc'],
            't': self.timer.t,
            'endpts': worms[0]['endpts'],
            'obs': obs,
            'reward': reward,
            'target': self.target,
            'action': action,
        }

        
    def reset(self,cam,task,target=None):
        """Returns the first `TimeStep` of a new episode."""
        if target is not None:
            self.target = target
        else:
            # Rotates target by 90 deg on each reset
            self.target = (self.target+90)%360
        task.write(0)
        self.bg = self.make_bgs(cam)

        # Get [window] locations to start proper HT tracking
        for i in range(self.window):
            img = grab_im(cam, self.bg)
            worms = find_worms(img, self.templates, self.bodies, ref_pts=[0,0], num_worms=1)
            if worms is None:
                self.recent_locs[i,:] = np.zeros((1,2))
            else:
                self.recent_locs[i,:] = worms[0]['loc']

        self.timer.reset()
        self.finished = False
        self.steps = 0
        
        # Takes one step and returns the observation
        self.head = self.recent_locs[-1,:] # doesn't have to be accurate; is just a starting point
        self.last_loc = self.recent_locs[-1,:]
        obs,reward,self.finished,info = self.step(0,cam,task)
        print('Done resetting\t')
        return obs
    
    def render(self, mode='human', close=False):
        pass

    """ BELOW: UTILITY FUNCTIONS """
    def make_bgs(self, cam):
        """Makes background images and stores in self"""
        return make_bg(cam,total_time=self.bg_time)
    
    
    def realobs2obs(self,realobs):
        # Takes angle measurements from -1 to 1 [theta_b, theta_h] and maps to obs [0,143]
        # EVENTUALLY THIS WILL BE FIXED TO PLAY NICE WITH THE MODEL ENV
        gridcoords = realobs*180
        # Below is from ensemble_mod_env utils
        if gridcoords[0]<-180 or gridcoords[0]>=180:
            if gridcoords[1]<-180 or gridcoords[0]>=180:
                raise ValueError('gridcoords are out of range.')
        tcoords = ((np.array(gridcoords)+180)/30).astype(int)
        return int(12*tcoords[0] + tcoords[1])