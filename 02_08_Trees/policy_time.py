'''
Compilation of functions needed for policy learning where the policy is informing data collection.
'''
import numpy as np
import pandas as pd

from improc import *
import utils as ut 
import worm_env as we 

class DataHandler():
    # Takes output of worm trials and formats/stores for model and agent use.
    # Tries to ensure that a matching parameters dictionary has been saved with each df. 
    def __init__(self, params=None):
        self.clear_df()
        if params:
            self.params = params

    def add_dict_to_df(self,fnames,**kwargs):
        # Takes fnames as a list.
        # kwargs are the arguments that would go to make_df() in utils.
        # Can run this when self.df is empty or exists. If exists, then adds to existing df.
        for key,val in kwargs.items():
            self.params[key] = val 
        if len(kwargs)==0:
            #print('No kwargs in add_dict_to_df')
            kwargs = self.params 
        for fname in fnames:
            self.df = ut.make_df(fname, old_frame=self.df, **kwargs)

    def add_df_to_df(self,fnames):
        # Takes fnames as list
        for fname in fnames:
            with open(fname,'rb') as f:
                self.df = self.df.append(pickle.load(f), ignore_index=True)

    def load_df(self,fname):
        with open(fname,'rb') as f:
            self.df = pickle.load(f)
        with open(fname[:-4]+'_params.pkl','rb') as f:
            self.params = pickle.load(f) 
        
    def clear_df(self):
        self.df = None 
        self.params = {
            'reward_ahead': None,
            'timestep_gap': None,
            'prev_act_window': None,
            'jump_limit': None,
        }

    def save_dfs(self,fname):
        self.df.to_pickle(fname) 
        with open(fname[:-4]+'_params.pkl','wb') as f:
            pickle.dump(self.params, f)
    
    def __str__(self):
        if self.df is None:
            return f'No dataframe\nParams are {self.params}'
        return f'Len of dataframe is {len(self.df)}\nParams are {self.params}'

def get_mod_and_policy(df, sm_pars=None, lp_frac=None):
    # Returns a model and policy.
    # The policy is in the form of probabilities for every state. A higher prob means
    # more preference to sample.
    mod, counts = ut.make_dist_dict2(df, sm_pars=sm_pars, lp_frac=lp_frac)
    counts = ut.smoothen(counts,counts,False,smooth_par=.1,iters=15)

    policy = np.exp(-np.abs(mod['reward_on'][:,:,0])/np.sqrt(mod['reward_on'][:,:,1]/counts))
    return mod, counts, policy

def transform_policy(policy, counts, threshold):
    # Taking a policy from the softmax of CoV and transforming it into something practical based on
    # worm state distributions.

    

'''
Worm functions
'''
def get_init_traj(fname, worm, episodes, act_rate=3, rand_probs=[.5,.5]):
    # act_rate is minimum amt of time passed before actions can change.
    cam,task = init_instruments()
    trajs = {}

    for i in range(episodes):
        worm.reset(cam,task)
        done = False
        t=0
        while done is False: 
            if t%act_rate == 0:
                action = np.random.choice([0,1],p=rand_probs)
            obs, rew, done, info = worm.step(action,cam,task)
            ut.add_to_traj(trajs, info)
            t+=1

    cam.exit()
    task.write(0)
    task.close()
    with open(fname,'wb') as f:
        pickle.dump(trajs,f)