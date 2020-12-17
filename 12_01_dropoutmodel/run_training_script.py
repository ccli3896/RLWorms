import multiprocessing 
import numpy as np

import model_based_agent as mba 
import worm_env as we 
import utils as ut
import tab_agents as tab
from datetime import datetime 

fbase = './Data/realworm_'+datetime.now().strftime('%d-%m-%H')+'_'

num_learners = 2
collection_eps = 10

def combine_learners(learners):
    # Just outputs averaged Q tables, so output is shape [144,2]
    output_shape = [144,2]
    averaged = np.zeros(output_shape)
    for lea in learners:
        averaged += (1/len(learners))*lea.Qtab  
    return averaged

def make_learner_list(dh, worm_pars={'num_models':1, 'frac':1}, **agentpars):
    learners = []
    for i in range(num_learners):
        agent = tab.Q_Alpha_Agent(**agentpars)
        learners.append(mba.Learner(agent, dh, 'a'+str(i), worm_pars=worm_pars))
    return learners

def main():

    # Start real worm environment
    worm = we.ProcessedWorm(0) 
    worm_agent = tab.Q_Alpha_Agent(gamma=0, epsilon=0.05, alpha=0) # Agent doesn't learn
    # Start model environments and learners
    dh = mba.DataHandler()
    dh.load_df('./nogap_traj_df.pkl')


    eps_vector = np.ones(10)
    eps_vector[::2] -= .95
    for loop in range(collection_eps):
        # Combining learners from previous run
        worm_agent.Qtab = combine_learners(learners)
        runner = WormRunner(worm, worm_agent)

        # Make new learner list based on newest data
        learners = make_learner_list(dh, gamma=0.25, epsilon=0.05, alpha=0.01)

        # Start multiprocessing
        manager = multiprocessing.Manager()
        poison_queue = manager.Queue()
        pool = multiprocessing.Pool()
        
        # Run main functions: train agents and collect more data
        fname = fbase+str(collection_eps)+'.pkl'
        pool.apply_async(runner.full_run, [1, fname], {eps_vector=eps_vector})
        for lea in learners:
            pool.apply_async(lea.learn, [], {'poison_queue':poison_queue})
        # Wait for them to finish
        pool.close()
        pool.join()

        # Take new data and add to DataHandler. 
        dh.add_dict_to_df(fname)

    # Save all new collected data plus old in one dataframe
    dh.save_dfs(fbase+'total.pkl')
    