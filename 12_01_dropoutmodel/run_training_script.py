import multiprocessing 
import numpy as np
import os

import model_based_agent as mba 
import worm_env as we 
#import fake_worm as fw
import utils as ut
import tab_agents as tab
from datetime import datetime 

folder = './Data/Reals'+datetime.now().strftime('%d-%m-%H-%M')
fbase = folder+'/realworm_'+datetime.now().strftime('%d-%m-%H-%M')+'_'
os.mkdir(folder)

num_learners = 2
collection_eps = 2

def combine_learners(learners):
    # Just outputs averaged Q tables, so output is shape [144,2]
    output_shape = [144,2]
    averaged = np.zeros(output_shape)
    for lea in learners:
        averaged += (1/len(learners))*lea.agent.Qtab  
    return averaged

def make_learner_list(dh, num_learners, worm_pars={'num_models':1, 'frac':.5}, **agentpars):
    learners = []
    for i in range(num_learners):
        agent = tab.Q_Alpha_Agent(**agentpars)
        learners.append(mba.Learner(agent, dh, 'a'+str(i), worm_pars=worm_pars))
    return learners

def main():
    '''
    At the end of this function, there will be files in a timestamped folder:
    1. Saved trajectory combined with old
    2. Saved individual trajectory files
    3. Averaged agent from each each full worm episode
    '''

    # Start real worm environment
    worm = we.ProcessedWorm(0) 
    worm_agent = tab.Q_Alpha_Agent(gamma=0, epsilon=0.05, alpha=0) # Agent doesn't learn
    # Start model environments and learners
    dh = mba.DataHandler()
    dh.load_df('./nogap_traj_df.pkl')
    learners = make_learner_list(dh, num_learners, gamma=0.25, epsilon=0.05, alpha=0.01)
    
    eps_vector = np.ones(2)
    eps_vector[::2] -= .95
    for loop in range(collection_eps):
        # Combining learners from previous run
        worm_agent.Qtab = combine_learners(learners)
        runner = mba.WormRunner(worm, worm_agent)

        # Make new learner list based on newest data
        learners = make_learner_list(dh, num_learners, 
                                gamma=0.25, epsilon=0.05, alpha=0.01, q_checkpoint=worm_agent.Qtab)

        # Start multiprocessing
        manager = multiprocessing.Manager()
        poison_queue = manager.Queue()
        pool = multiprocessing.Pool()
        
        # Run main functions: train agents and collect more data
        fname = fbase+str(collection_eps)+'.pkl'
        lea_outs = []
        for lea in learners:
             lea_outs.append(pool.apply_async(lea.learn, [], {'poison_queue':poison_queue}))
        runner_out = pool.apply_async(runner.full_run, [2, fname], {'eps_vector':eps_vector,'poison_queue':poison_queue})
        # Wait for them to finish
        pool.close()
        pool.join()
        
        # Take new data and add to DataHandler. 
        dh.add_dict_to_df([fname])

    runner.close()
    # Save all new collected data plus old in one dataframe
    dh.save_dfs(fbase+'total.pkl')
    return lea_outs, runner_out
    
if __name__=='__main__':
    lea_outs, runner_out = main()