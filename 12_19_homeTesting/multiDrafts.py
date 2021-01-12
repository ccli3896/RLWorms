'''
Script that's just for functions needed for multiprocessing.
(Scratchwork and drafts)
'''

import model_based_agent as mba 
import numpy as np 


def no_learn_performance(samples,frac):
    # Returns the performance of the eval episode using a policy based on differences
    # between reward_on and reward_off matrices.
    # Also returns the q matrix differences.
    learner, dh = one_learner(frac=frac)
    dh.df = dh.df[:samples]
    learner.make_mod_and_env(dh,{'lambda':.1,'iters':10})
    q_diffs = learner.modset.models[0]['reward_on'][:,:,0]-learner.modset.models[0]['reward_off'][:,:,0]
    
    learner,dh = one_learner(frac=1)
    learner.make_mod_and_env(dh,{'lambda':.1,'iters':10})
    
    # Artificially set Q table
    learner.agent.Qtab[:,0] = np.zeros(144)
    learner.agent.Qtab[:,1] = np.sign(q_diffs.flatten())
    
    # Test on full model
    obss,acts,rews = eval_ep(learner)
    perf = np.mean(rews)
    
    return perf, q_diffs

def one_learner(
    num_learners = 1,
    collection_eps = 1,
    gamma=0.25,
    epsilon=.5,
    alpha=.001,
    df_bound=-1,
    frac=1,
):

    qtabs = []
    # Start model environments and learners 
    dh = mba.DataHandler()
    dh.load_df('ensemble_testing.pkl')
    dh.df = dh.df[:df_bound]

    # Make new learner list based on newest data
    learners = mba.make_learner_list(num_learners, worm_pars={'num_models':1, 'frac':frac},
                            gamma=gamma, epsilon=epsilon, alpha=alpha)  
    return learners[0],dh

def eval_ep(lea,eval_steps=200000):
    obss = []
    acts = []
    rews = []
    for i in range(eval_steps):
        obs = lea.env._state
        acts.append(lea.agent.eval_act(obs))
        next_obs,rew,done,_ = lea.env.step(acts[-1])
        obss.append(next_obs)
        rews.append(rew)
    return obss,acts,rews