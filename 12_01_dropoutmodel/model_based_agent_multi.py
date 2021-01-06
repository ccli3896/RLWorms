import numpy as np 
import utils as ut 
import pickle
import ensemble_mod_env as eme
import fake_worm as fw
import worm_env as we

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

class Learner():
    # Agents take a model_set object and sample randomly, uniformly, from all models at each step.
    # The agent_manager trains one agent on model_set in separate processes. Means multiple agent_managers need
    # to be spawned to learn in parallel.

    def __init__(self, agent, handler, label, worm_pars={'num_models':10, 'frac':.5}, num_steps=1000,eval_steps=1000):
        # Stores an agent.
        self.agent = agent 
        self.label = label
        self.reset_reward_vecs()

        self.num_steps = num_steps
        self.eval_steps = eval_steps
        
        # Make different sampled group of worms for each Learner object
        modset = eme.ModelSet(worm_pars['num_models'], frac=worm_pars['frac'])
        modset.make_models(handler,sm_pars={'lambda':.05, 'iters':30})
        self.env = eme.FakeWorm(modset)

    def reset_reward_vecs(self):
        self.rewards = []
        self.eval_rewards = []

    def _learn_step(self):
        # Internal function to take one step.
        # Chooses an action based on model state, takes action, gets info. 
        # Updates.
        obs = self.env._state
        action = self.agent.act(obs)
        next_obs,rew,done,_ = self.env.step(action)
        self.agent.update(obs,action,next_obs,rew)  
        return rew
    
    def _eval_step(self):
        obs = self.env._state 
        action = self.agent.eval_act(obs) 
        next_obs,rew,done,_ = self.env.step(action)
        return rew
    
    def eval_ep(self):
        self.eval_rewards = []
        for _ in range(self.eval_steps):
            self.eval_rewards.append(self._eval_step())
        return self.eval_rewards

    def learn(self,poison_queue=None,learn_limit=None):
        # Learning loop. Has an option for a poison_queue input, which will stop and return the function
        # if a stop signal is received.

        # TODO: rewrite poison_queue to use .get() instead of this mess
        learn_eps = 0
        empty_poison_queue = True

        while empty_poison_queue and (learn_limit is None or learn_eps<learn_limit):
            if poison_queue is None: 
                empty_poison_queue = True
            else: 
                empty_poison_queue = poison_queue.empty()

            for i in range(self.num_steps):
                self.rewards.append(self._learn_step())
            learn_eps+=1

        # After the learner gets a signal to stop, do an eval episode
        self.env.reset()
        return learn_eps, np.mean(self.eval_ep())
        
    def save_agent(self,fname):
        with open(fname,'wb') as f:
            pickle.dump(self.agent, f)
    
    def save_rewards(self,fname):
        with open(fname,'wb') as f:
            pickle.dump(self.rewards, f)
        with open(fname[:-4]+'_eval.pkl','wb') as f:
            pickle.dump(self.eval_rewards, f)

class WormRunner():
    # Can run multiple types of worm episodes. Each must have the option to return a stop code that plays nice with
    # agent_manager and pool.apply_async(). 

    def __init__(self,agent,worm):
        # Start a worm (worm is the env ProcessedWorm)
        # agent should be an agent class with averaged qtables from a multiprocessed run. 
        self.worm = worm
        self.agent = agent
        self.traj = {}
        self.eval_traj = {}

    def eval_ep(self,fname):
        # Runs an evaluation episode on the worm and returns the total rewards collected.
        self.eval_traj = {}
        obs = self.worm.realobs2obs(self.worm.reset())
        done = False
        while not done:
            action = self.agent.eval_act(obs)
            next_obs, rew, done, info = self.worm.step(action, sleep_time=0) 
            self.add_to_traj(self.eval_traj, info)
            obs = self.worm.realobs2obs(next_obs)
        with open(fname,'wb') as f:
            pickle.dump(self.eval_traj, f)
        return np.mean((self.eval_traj['reward']))

    def full_run(self, num_eps, fname='tester.pkl', eps_vector=None, poison_queue=None):
        # Runs a number of episodes of runtype. Saves trajectory at end.
        # eps_vector is a list of epsilon values for each episode. If there's one value [eps], then 
        # every episode uses the same value.
        # poison_queue is a multiprocessing.Manager.Queue() object
        #
        # Saves traj 
        if eps_vector is None:
            eps_vector = np.zeros(num_eps)+.1
        elif len(eps_vector)==1:
            eps_vector = np.zeros(num_eps)+eps_vector

        self.eval_ep(fname[:-4]+'_eval_start.pkl')
        self.traj = {} 
        for ep in range(num_eps):
            self.eps_greedy_ep(eps_vector[ep]) # Background is reset each time this is called
        self.save_traj(fname) 
        self.eval_ep(fname[:-4]+'_eval_end.pkl')

        if poison_queue is not None:
            poison_queue.put('STOP')

    def eps_greedy_ep(self,epsilon):
        self.agent.epsilon = epsilon
        obs = self.worm.realobs2obs(self.worm.reset())
        done = False
        while not done:
            action = self.agent.act(obs)
            next_obs, rew, done, info = self.worm.step(action, sleep_time=0)
            self.add_to_traj(self.traj, info)
            obs = self.worm.realobs2obs(next_obs) 

    def save_traj(self,fname):
        with open(fname,'wb') as f:
            pickle.dump(self.traj, f)

    def close(self):
        self.worm.close()

    def add_to_traj(self,trajectory,info):
        # appends each key in info to the corresponding key in trajectory.
        # If trajectory is empty, returns trajectory as copy of info but with each element as list
        # so it can be appended to in the future.

        if trajectory:
            for k in info.keys():
                trajectory[k].append(info[k])
        else:
            for k in info.keys():
                trajectory[k] = [info[k]]

#############################
# To dos
#############################
'''
Notes on implementation.
The learner and worm_runner will be running with pool.apply_async(), and agent_manager will go until worm_runner 
is done for each episode. That means worm_runner needs a closed data-collection method that can be sent to a process.
Also means that agent_manager needs a short method that runs a number of training steps, looping for as long as 
worm_runner goes. Poisonpill method:
https://stackoverflow.com/questions/29571671/basic-multiprocessing-with-while-loop
Thinking this with a poison_pill argument for when worm_runner is done could work. As in, run a loop inside the 
agent_manager and while queue is empty, keep training. When queue receives a poisonpill, close up and exit.
'''