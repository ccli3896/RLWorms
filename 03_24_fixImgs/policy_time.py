'''
Compilation of functions needed for policy learning where the policy is informing data collection.
'''
import numpy as np
import pandas as pd
from scipy.stats import norm

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
        self.df = ut.make_df(fnames, old_frame=self.df, **kwargs)

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
        
    def clear_df(self,params=None):
        self.df = pd.DataFrame()
        if params is None:
            self.params = {
                'reward_ahead': None,
                'timestep_gap': None,
                'prev_act_window': None,
                'jump_limit': None,
            }
        else:
            self.params = params

    def save_dfs(self,fname):
        self.df.to_pickle(fname) 
        with open(fname[:-4]+'_params.pkl','wb') as f:
            pickle.dump(self.params, f)
    
    def __str__(self):
        if len(self.df)==0:
            return f'No dataframe\nParams are {self.params}'
        return f'Len of dataframe is {len(self.df)}\nParams are {self.params}'

'''
Utilities for dataframes
'''
def change_reward_ahead(df,reward_ahead,jump_limit=20):
    # Takes a dataframe where reward_ahead setting was 1.
    # Rewrites reward column using new reward_ahead
    # Returns dataframe at the end.
    start_inds = [0]
    for i in range(len(df)-1):
        if pt_dist(df['loc'][i],df['loc'][i+1]) > jump_limit:
            start_inds.append(i+1)
    start_inds.append(len(df))

    new_df = pd.DataFrame(columns=df.columns)
    for i in range(len(start_inds)-1):
        new_sec = df.iloc[start_inds[i]:start_inds[i+1]-reward_ahead].copy()
        new_r = [np.sum(df['reward'][start_inds[i]+j:start_inds[i]+j+reward_ahead]) 
                    for j in range(len(new_sec))]
        new_sec['reward'] = new_r
        new_df = new_df.append(new_sec)
    return new_df

'''
Policy construction
'''
def entropy(probs):
    return -probs*np.log(probs+1e-6)-(1-probs)*np.log(1-probs+1e-6)
def POver0(mu,sig):
    # Returns the probability that the RV is above 0
    # Lambda function returns elementwise
    return 1-norm.cdf(-mu/sig)

def bart2pols(posterior,posterior_sig):
    # Takes posterior sampling from BART in R (size [n_samples,288]) where
    # actions alternate every entry. Then second variable ascends fastest.
    # posterior_sig is also returned from BART and is [n_samples].
    # Returns the array of P(Q_delta(state)>0) samples for each state based on posterior and posterior_sig.
    # Also returns those probabilities turned to entropies (can be used in TS directly).
    # Both shapes are [n,144]
    
    post_diff = posterior[:,1::2] - posterior[:,::2]
    post_sig = np.repeat(posterior_sig.reshape(-1,1),144,axis=1)
    post_sig = np.sqrt(np.square(post_sig)*2)
    # Now post_diff is shaped like [n,144] where n is posterior sample num
    probs = POver0(post_diff,post_sig)
    ents = entropy(probs)
    return ents,probs

def get_counts(df):
    # Gets counts for each entry
    df_view = df.groupby(['obs_b','obs_h'])['reward'].agg(['mean','count'])
    df_view.reset_index(inplace=True)
    counts = np.zeros((12,12))
    for i,ob in enumerate(np.arange(-180,180,30)):
        for j,oh in enumerate(np.arange(-180,180,30)):
            if ((df_view['obs_b'] == ob) & (df_view['obs_h'] == oh)).any():
                counts[i,j] = df_view[(df_view['obs_b']==ob)&(df_view['obs_h']==oh)]['count']
    return counts

def thompson_sampling(entropies, counts, light_limit):
    # Takes entropies and ranks them for Thompson sampling. They are in shape [n,144].
    # Form sampling probability matrix, adjusting for light_limit and counts 
    #   (prob of being in each state)
    # Returns PROBABILITY LIGHT IS ON, NOT SAMPLING PROBABILITY 
    # as in, p_light_on = .5*p_sample

    # Get sorted entropies 
    counts = counts.flatten()/np.sum(counts) + 1e-6 # get counts in same format as ents
    sts = len(counts) # number of states

    # Make p:
    # The first row is the proportion each state came in first;
    # the second row is the proportion it came in second, etc.
    # Each column is the proportions for that state.
    p = np.zeros((sts,sts)) 
    for i in range(sts):
        p[i,:], _ = np.histogram(np.argsort(-entropies,axis=1)[:,i],
            bins=range(sts+1),density=True)

    pol = np.zeros(sts)
    rank = 0
    for i in range(sts):

        # If all states in first row are used up but light isn't yet
        while np.sum(p[rank,:])==0:
            rank+=1
        best = np.argmax(p[rank,:])
        pol[best] = p[rank,best]/np.sum(p[rank,:])*light_limit 

        # Check if policy exceeds state occupancy
        if pol[best] > counts[best]:
            pol[best] = counts[best]
        
        light_limit -= pol[best]
        p[:,best] = 0

    pol /= (counts*2)
    pol = np.minimum(pol,.5)

    return pol.reshape(12,12)

def make_sprobs_cutoff(ents,alpha,counts,baseline):
    # Takes the entropies and returns a matrix of action probabilities for a=1 (light on).
    # This is the case where we only sample top and randomly sample otherwise (greedy). 
    # 
    # ents is the matrix of entropies.
    # alpha is the sampling limit. Goes from 0 to 1. A 1 means light is on 50% of time.
    # counts is how many times each state was visited.
    # baseline is the minimum sampling rate for all states. 

    dist = counts/np.sum(counts)

    # Index the distribution by sorted entropies. Then find the number of terms whose cumulative sum
    # is just less than alpha.
    top_n = np.sum(np.cumsum(dist.ravel()[np.argsort(-ents.ravel())])<=((alpha-baseline)/(1-baseline+1e-6)))
    samp_inds = np.dstack(np.unravel_index(np.argsort(-ents.ravel())[:top_n], ents.shape)).reshape(-1,2)
    
    # Construct the sampling probability matrix
    sprobs = np.zeros(dist.shape)+baseline
    for s in samp_inds:
        sprobs[s[0],s[1]] = 1
    return sprobs*.5

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

def do_sampling_traj(probs, fname, worm, episodes, act_rate=3):
    # probs is the probability that the light will be on in a state.
    # min 0, max .5. 
    # fname is the file the trajectory will be saved in, a pkl.
    # worm is worm object as usual
    # episodes is number of worm obj episodes to run
    # act_rate is min steps passed before actions can change.
    def obs2ind(obs):
        # Turns obs with entries range [-1,1] to [0,11]
        return np.round(obs*6+6).astype(int)

    cam,task = init_instruments()
    trajs = {}

    for i in range(episodes):
        worm.reset(cam,task)
        done = False
        t,action = 0,0
        while done is False:
            obs, rew, done, info = worm.step(action,cam,task) 
            ut.add_to_traj(trajs, info)

            # Choose action based on probabilities
            if np.sum(info['endpts'])==-4:
                # No worm found
                action = 0
            elif t%act_rate == 0:
                # Get prob of light on.
                ind = obs2ind(obs)
                prob = probs[ind[0],ind[1]]
                action = np.random.choice([0,1],p=[1-prob, prob])
            
            t+=1

    cam.exit()
    task.write(0)
    task.close()
    with open(fname,'wb') as f:
        pickle.dump(trajs,f)