'''
The main policy-running script.
get_posterior_bart.R needs to be running in the background.
'''

import numpy as np
import pandas as pd
import pickle 
import os 

from improc import *
import utils as ut 
import policy_time as pt 
import worm_env as we 


def gen_pol_collection(
    worm_id,
    folder, 
    episodes = 10,
    ep_len = 3*60, # episode in seconds
    init_eps = 1, # number of randomly sampled episodes to start out with
    light_limit = .2, # max proportion of time we want light to be on
    state_space_shape = (12,12), # just in case this changes
):

    # Fixed parameters
    params = {
        'reward_ahead': 30,
        'timestep_gap': 1, 
        'prev_act_window': 3,
        'jump_limit': 20,
    }
     
    # Initialize objects 
    worm = we.ProcessedWorm(0, ep_len=ep_len)
    dh = pt.DataHandler(params=params)
    trajnames = []   

    for ep in np.arange(episodes)+worm_id*episodes:

        # If we're still in the inits
        if (worm_id==0) and (ep in range(init_eps)):
            sampling_probs = np.zeros(state_space_shape) + light_limit
        elif (worm_id!=0) and (ep%episodes==0):
        # We're not in the inits anymore and need to load the policy
            sampling_probs = np.load(f'{folder}pol{ep-1}.npy')
            dh.load_df(f'{folder}fulltraj.pkl')

        # Get new sample 
        trajname = f'{folder}traj{ep}.pkl' # dict form
        pt.do_sampling_traj(sampling_probs, trajname, worm, 1)

        # Update dataframe and save
        dh.add_dict_to_df([trajname])
        dh.df = dh.df[dh.df['prev_actions'].isin([0,3])] # Only keep usable points
        dh.save_dfs(f'{folder}fulltraj.pkl')

        # Send to BART and wait til it gets back to me
        pt.save_for_R(dh,f'{folder}traj{ep}.npy')
        bart_f = f'{folder}bart{ep}.npy'
        bart_f_s = f'{folder}bartsig{ep}.npy'
        while not os.path.exists(bart_f) or not os.path.exists(bart_f_s):
            # Just in case there's a tiny difference between the two in time
            time.sleep(1)

        # Get probs and save them (can transform to entropy later)
        post = np.load(bart_f, allow_pickle=True)
        postsig = np.load(bart_f_s, allow_pickle=True)
        ents, probs = pt.bart2pols(post,postsig)
        with open(f'{folder}probs{ep}.pkl','wb') as f:
            pickle.dump(probs,f)
        with open(f'{folder}ents{ep}.pkl','wb') as f:
            pickle.dump(ents,f)

        # Get counts 
        counts = pt.get_counts(dh.df)
        sampling_probs = pt.thompson_sampling(ents,counts,light_limit)
        np.save(f'{folder}pol{ep}.npy', sampling_probs) 