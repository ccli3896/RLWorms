'''
Compilation of functions needed for policy learning where the policy is informing data collection.
'''
import numpy as np
import pandas as pd

from improc import *
import utils as ut 

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

'''
Utilities for dataframes
'''
def change_reward_ahead(df,reward_ahead,jump_limit=100):
    # Takes a dataframe where reward_ahead setting was 1.
    # Rewrites reward column using new reward_ahead
    # Returns dataframe at the end.
    start_inds = [0]
    for i in range(len(df)-1):
        if pt_dist(df['loc'][i],df['loc'][i+1]) > 10:
            start_inds.append(i+1)

    new_df = pd.DataFrame(columns=df.columns)
    for i in range(len(start_inds)-1):
        new_sec = df.iloc[start_inds[i]:start_inds[i+1]-reward_ahead].copy()
        new_r = [np.sum(dh.df['reward'][start_inds[i]+j:start_inds[i]+j+reward_ahead]) 
                     for j in range(len(new_sec))]
        new_sec['reward'] = new_r
        new_df = new_df.append(new_sec)
    return new_df

'''
R interaction functions
'''

def save_for_R(df,fname):
    ang2r = lambda x: x/30+7

    obs_b_on = ang2r(df[df['prev_actions']==3]['obs_b'].to_numpy())
    obs_b_off = ang2r(df[df['prev_actions']==0]['obs_b'].to_numpy())
    obs_h_on = ang2r(df[df['prev_actions']==3]['obs_h'].to_numpy())
    obs_h_off = ang2r(df[df['prev_actions']==0]['obs_h'].to_numpy())
    rew_on = df[df['prev_actions']==3]['reward'].to_numpy()
    rew_off = df[df['prev_actions']==0]['reward'].to_numpy()

    ons = [obs_b_on,obs_h_on,rew_on]
    offs = [obs_b_off,obs_h_off,rew_off]

    ons = [np.expand_dims(on,1) for on in ons]
    offs = [np.expand_dims(off,1) for off in offs]
    ons = np.concatenate(ons,axis=1)
    offs = np.concatenate(offs,axis=1)

    alls = np.concatenate([ons,offs],axis=0)
    acts = np.zeros((alls.shape[0],1))
    acts[:ons.shape[0]] += 1
    alls = np.concatenate([alls,acts],axis=1)
    alls = alls.astype(float)
    
    np.save(fname, alls, allow_pickle=True)
    print("Saved as numpy for R")

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