import multiprocessing 
import numpy as np

import model_based_agent as mba 
import worm_env as we 
import utils as ut
import tab_agents as tab

poison_pill = 'STOP'

def main():
    # Import starting data collected earlier, random.
    dh = mba.DataHandler()
    dh.load_df('./nogap_traj_df.pkl')

    # 
    worm = we.ProcessedWorm(0) 
    learners = []
    for i in range(num_learners):
        agent = tab.Q_Alpha_Agent(gamma=0.5,
                                epsilon=0.05,
                                alpha=0.005,
                                )
        learners.append(mba.Learner(agent, dh, 'a'+str(i))) 

     