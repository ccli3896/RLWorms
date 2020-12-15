import numpy as np 
import matplotlib.pyplot as plt 
import utils as ut 
import model_env as me 
import worm_env as we 
import pickle
import pandas as pd 
import tab_agents as tab 

'''
Flow of script:
Take previously saved trajectory and make several models by sampling data.
Have multiple agents learning on the models
RANDOM POLICY: Simultaneously collect more data from a worm. 

Consolidate new trajectory with old and make several more models.
Have multiple agents learning on the models
EP-GREEDY POLICY: Simultaneously collect more data from a worm following an epsilon-greedy
    policy from averaged previous agents. 

Repeat above two blocks indefinitely
'''

class DataHandler():
    # Takes output of worm trials and formats/stores for model and agent use.
    # Tries to ensure that a matching parameters dictionary has been saved with each df. 
    def __init__(self):
        self.clear_df()

    def add_dict_to_df(self,fnames,**kwargs):
        # Takes fnames as a list.
        # kwargs are the arguments that would go to make_df() in utils.
        # Can run this when self.df is empty or exists. If exists, then adds to existing df.
        for key,val in kwargs.items():
            self.params[key] = val 
        if len(kwargs)==0:
            print('No kwargs')
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

 
class model_set():
    # Creates and stores lists of models that sample randomly from saved trajectories.
    # Idea for now: each model is actually a large list of sampled models.
    def __init__(self,num_models,samples):
        pass 
    def update_df(self):
        pass
    def get_model(self):
        # Makes distribution dictionary for a model.
        pass 


class learner():
    # Agents take a model_set object and sample randomly, uniformly, from all models at each step.
    # The agent_manager trains one agent on model_set in separate processes. Means multiple agent_managers need
    # to be spawned to learn in parallel.
    def __init__(self):
        # Makes an agent, inits hps. 
        pass 
    def _learn_step(self):
        # Internal function to take one step.
        pass 
    def learn(self,poison_queue=None):
        # Learning loop. Has an option for a poison_queue input, which will stop and return the function
        # if a stop signal is received.
        pass

class learner_manager():
    # 

class worm_runner():
    # Can run multiple types of worm episodes. Each must have the option to return a stop code that plays nice with
    # agent_manager and pool.apply_async(). 
    def __init__(self):
        pass 
    def eps_greedy_run(self):
        pass
    def boltzmann_run(self):
        pass
    def random_run(self):
        pass 

#############################
# To dos
#############################
'''
Notes on implementation.
The agent_manager and worm_runner will be running with pool.apply_async(), and agent_manager will go until worm_runner 
is done for each episode. That means worm_runner needs a closed data-collection method that can be sent to a process.
Also means that agent_manager needs a short method that runs a number of training steps, looping for as long as 
worm_runner goes. Poisonpill method:
https://stackoverflow.com/questions/29571671/basic-multiprocessing-with-while-loop
Thinking this with a poison_pill argument for when worm_runner is done could work. As in, run a loop inside the 
agent_manager and while queue is empty, keep training. When queue receives a poisonpill, close up and exit.
'''