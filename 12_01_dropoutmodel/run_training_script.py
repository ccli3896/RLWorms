import multiprocessing 
import numpy as np
import os

import model_based_agent as mba 
import worm_env as we 

import utils as ut
import tab_agents as tab
from datetime import datetime 

folder = './Data/Reals'+datetime.now().strftime('%d-%m-%H-%M')
fbase = folder+'/realworm_'
if os.path.isdir(folder):
    os.path.rmdir(folder)
os.mkdir(folder)

num_learners = 5
collection_eps = 5
gamma=.25
epsilon=.05
alpha=.01


if __name__=='__main__':
    
    '''
    At the end of this script, there will be files in a timestamped folder:
    1. Saved trajectory combined with old ('..total.pkl')
    2. Saved individual trajectory files ('..eval_start.pkl', '..[ep].pkl')
    3. Averaged agent from each each full worm episode ('..[ep]_agent.pkl')
    '''

    # Start real worm environment
    worm = we.ProcessedWorm(0,ep_len=1200,bg_time=20) 
    worm_agent = tab.Q_Alpha_Agent(gamma=0, epsilon=0.05, alpha=0) # Agent doesn't learn
    # Start model environments and learners 
    dh = mba.DataHandler()
    dh.load_df('./nogap_traj_df.pkl')
    lea_outs = []
    learners = mba.make_learner_list(dh, num_learners, gamma=gamma, epsilon=epsilon, alpha=alpha)
    
    eps_vector = np.ones(2)
    eps_vector[::2] -= .95
    for loop in range(collection_eps):
        fname = fbase+str(loop)+'.pkl'
        
        # Combining learners from previous run
        if len(lea_outs)==0:
            averaged = np.zeros((144,2))
            for lea in learners:
                averaged += (1/len(learners))*lea.agent.Qtab
        else:
            worm_agent.Qtab = mba.combine_learners(lea_outs)
        worm_agent.save(fname[:-4]+'_agent.pkl')

        # Make new learner list based on newest data
        learners = mba.make_learner_list(dh, num_learners, 
                                gamma=gamma, epsilon=epsilon, alpha=alpha, q_checkpoint=worm_agent.Qtab)

        # Start multiprocessing
        manager = multiprocessing.Manager()
        poison_queue = manager.Queue()
        pool = multiprocessing.Pool()
        
        # Run main functions: train agents and collect more data
        lea_outs = []
        for lea in learners:
            print('learning')
            lea_outs.append(pool.apply_async(lea.learn, [],{'learn_limit':5000}))
        # Wait for them to finish
        pool.close()
        pool.join()
        runner = mba.WormRunner(worm_agent,worm)
        runner.full_run(2, fname, eps_vector=eps_vector)
        
        # Take new data and add to DataHandler. 
        dh.add_dict_to_df([fname])

    runner.close()
    # Save all new collected data plus old in one dataframe
    dh.save_dfs(fbase+'total.pkl')