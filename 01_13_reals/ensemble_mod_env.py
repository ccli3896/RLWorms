import gym
from gym import spaces
import numpy as np
import utils as ut

class FakeWorm(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, modset, model_id=None, start=None, ep_len=None):
        '''
        modset is a ModelSet object. Sampling will be done from it. 
        This environment does not modify it.
        If model_id is not set, then it'll sample from different dist_dicts every time.
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
        self.state_inds = self.grid2coords(self.state)
            
        # Setting self parameters
        self.modset = modset
        self.model_id = model_id
        self.ep_len = ep_len
        self.steps = 0
        self.finished = False
    

    def step(self, action):
        """Chooses action and returns a (step_type, reward, discount, observation)"""
        self.steps += 1
        # If step count reaches episode length
        if self.ep_len is not None and self.steps >= self.ep_len:
            self.finished = True
            
        # Draws new reward and state from previous state 
        self.state[0],self.state[1],reward = self.modset.sample(self.state_inds,action,model_id=self.model_id)
                       
        # Return obs, reward, done (boolean), info (dict)
        self._state = self.grid2obs(self.state)
        self.state_inds = self.grid2coords(self.state)
        return self._state, reward, self.finished, {}

        
    def reset(self,model_id=None,start=None):
        """Returns the first `TimeStep` of a new episode."""
        self.finished = False
        self.model_id = model_id
        self.steps = 0
        # Setting initial conditions
        if start is None:
            self._state = np.random.choice(self.grid_width**2)
        else:
            self._state = self.grid2obs(start)
        self.state = self.obs2grid(self._state)
        self.state_inds = self.grid2coords(self.state)
        
        # Takes one step and returns the observation
        return self._state
    
    def render(self, mode='human', close=False):
        print('At',self.state)              

    ''' Conversion functions '''
    # obs is from 0 to 143.
    # grid is [-180 to 150, -180 to 150]
    # coords is [0 to 11, 0 to 11]
    def obs2grid(self,obs):
        if obs<0 or obs>143:
            raise ValueError('obs is out of range.')
        gridcoords = np.array([obs//self.grid_width, obs%self.grid_width])
        return (gridcoords*30)-180
    def grid2obs(self,gridcoords):
        if gridcoords[0]<-180 or gridcoords[0]>=180:
            if gridcoords[1]<-180 or gridcoords[0]>=180:
                raise ValueError('gridcoords are out of range.')
        tcoords = ((np.array(gridcoords)+180)/30).astype(int)
        return self.grid_width*tcoords[0] + tcoords[1]
    def grid2coords(self,gridcoords):
        if gridcoords[0]<-180 or gridcoords[0]>=180:
            if gridcoords[1]<-180 or gridcoords[0]>=180:
                raise ValueError('gridcoords are out of range.')
        coords = (np.array(gridcoords)+180)//30
        return coords
    def coords2grid(self,coords):
        #Unused
        return (np.array(coords)*30)-180
    def obs2coords(self,obs):
        if obs<0 or obs>143:
            raise ValueError('obs is out of range.')
        return np.array([obs//self.grid_width, obs%self.grid_width])
    ''' End of conversion functions '''

class ModelSet():
    # Creates and stores lists of models that sample randomly from saved trajectories.
    # Idea for now: each model is actually a large list of sampled models. The dataframe is only used to sample
    # from at first and isn't stored in this object.
    # Lp_frac: [0,1]. Models find a number to subtract from light-on matrices, in order for
    #  this fraction of observations to remain above their corresponding light-off spots.
    #  Implemented in utils.make_dist_dict.

    def __init__(self,num_models,samples=None,frac=None,lp_frac=0.5):
        self.num_models = num_models
        self.frac = frac 
        self.samples = samples
        self.models = []
        self.model_params = None
        self.lp_frac=lp_frac

    def make_models(self,handler,sm_pars):
        # handler is a DataHandler object from model_based_agent file.
        # sm_pars: {lambda, iters}
        self.model_params = dict(handler.params) 
        self.model_params['sm_pars'] = sm_pars 
        self.models = []
        for _ in range(self.num_models):
            print(f'On model {_}')
            if self.frac is not None:
                samps = handler.df.sample(frac=self.frac)
            elif self.samples is not None:
                samps = handler.df.sample(n=self.samples)
            else:
                raise ValueError('Samples and frac cannot both be None')

            self.models.append(ut.make_dist_dict2(samps, sm_pars=sm_pars, 
                                prev_act_window=self.model_params['prev_act_window'],
                                lp_frac=self.lp_frac))
    
    def sample(self,inds,action,model_id=None):
        # inds is the list to say which states to sample for. [theta_b,theta_h]
        # In format coords.
        # model_id is if we want to sample consistently from one model. 
        # Returns state and reward values. [body, head, reward]
        if model_id is None:
            model_id = np.random.choice(self.num_models)
        
        if action==0:
            keystrs = ['body_off','head_off','reward_off']
        elif action==1:
            keystrs = ['body_on','head_on','reward_on']
        else:
            raise ValueError('Invalid action')
    
        next_step = []
        for i in range(3):
            mu,var = self.models[model_id][keystrs[i]][inds[0],inds[1],:]
            if i<2:
                next_step.append(ut.wrap_correct(ut.myround(np.random.normal(mu,np.sqrt(var))),buffer=180))
            else:
                next_step.append(np.random.normal(mu,np.sqrt(var)))
        
        return next_step 
        
    
    def __str__(self):
        return f'ModelSet contains {len(self.models)} \nFrac {self.frac}, Samples {self.samples}\nParams {self.model_params}'