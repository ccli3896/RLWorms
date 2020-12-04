import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys

import bin_utils as ut
import bin_model_env as me
import tab_agents as tab


'''
Setting hyperparameters
'''
i = int(sys.argv[1])
alphas = [.1, .05, .01, .005, .001]
gammas = [.7, .8, .9, .95, .98]
epsilons = [.1, .05, .01]
runtimes = [10000, 20000]

it = i//150
i -= it*150
alpha_ind = i//30
gamma_ind = (i-(30*alpha_ind))//6
epsilon_ind = (i-30*alpha_ind-6*gamma_ind)//2
runtime_ind = (i-30*alpha_ind-6*gamma_ind-2*epsilon_ind)

alpha = alphas[alpha_ind]
gamma = gammas[gamma_ind]
epsilon = epsilons[epsilon_ind]
runtime = runtimes[runtime_ind]

fbase = './Outputs/Bin1_a'+str(alpha_ind)+'g'+str(gamma_ind)+'e'+str(epsilon_ind)+'r'+str(runtime_ind)+'_iter'+str(it)+'.json'

'''
Making dist dict
'''

fnames=[
    # First worm
    'Data/traj12-11-2020_19-04-41.pkl', #none
    'Data/traj12-11-2020_19-14-38.pkl', #none
    'Data/traj12-11-2020_19-24-30.pkl', #xlim 800
    # 'Data/traj12-11-2020_19-35-31.pkl', #none # Seems like an especially bad dataset. Actually ruined all the others
    # Second worm
    'Data/traj12-11-2020_19-55-19.pkl', #none
    'Data/traj12-11-2020_20-05-11.pkl', #none
    'Data/traj12-11-2020_20-15-17.pkl', #none
    'Data/traj12-11-2020_20-25-06.pkl', #xlim 1430
]

xlims = [1e6,1e6,800,1e6,1e6,1e6,1430]

traj_df = ut.make_df(fnames,xlimit=xlims,time_steps=10)
dist_dict = ut.make_dist_dict(traj_df, bin_z=1)


'''
Running the script 
'''
worm = me.FakeWorm(dist_dict)
alph_mouse = tab.Q_Alpha_Agent(worm,gamma=gamma,epsilon=epsilon,alpha=alpha)

alpha_mouse_learned, rewards, eval_rewards = tab.learner(alph_mouse,worm,episodes=runtime)

alpha_mouse_learned.rewards = rewards
alpha_mouse_learned.eval_rewards = eval_rewards
with open(fbase,'wb') as f:
    pickle.dump(alpha_mouse_learned,f)

