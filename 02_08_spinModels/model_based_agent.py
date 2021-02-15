import numpy as np 
import utils as ut 
import pickle
import ensemble_mod_env as eme

import worm_env as we
import tab_agents as tab
from improc import *

'''
This is all written for multiprocessing.
Major differences: WormRunner has to make its own cam and task in the full_run method.
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
            #print('No kwargs in add_dict_to_df')
            kwargs = self.params 
        for fname in fnames:
            self.df = ut.make_df(fname, old_frame=self.df, **kwargs)

    #def add_dict_to_df_HT(self,fnames,**kwargs):
        # Difference between this and above fn is that post-processing is done on HT angles
        # to ensure continuity. Trajectory must have keys 't', 'obs_b', 'angs', 'prev_actions', 'reward', 'loc',
        # 'target
        # TODO 

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

    def __init__(self, agent, label, worm_pars={'num_models':10, 'frac':.5}, 
        num_steps=100,eval_steps=100000,
        lp_frac=1/3):
        # Stores an agent.
        self.agent = agent 
        self.label = label
        self.reset_reward_vecs()

        self.num_steps = num_steps
        self.eval_steps = eval_steps
        self.lp_frac = lp_frac
        
        # Make different sampled group of worms for each Learner object
        self.modset = eme.ModelSet(worm_pars['num_models'], frac=worm_pars['frac'], lp_frac=self.lp_frac)

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
    
    def make_mod_and_env(self,handler,sm_pars):
        self.modset.make_models(handler,sm_pars=sm_pars)
        self.env = eme.FakeWorm(self.modset)
    
    def eval_ep(self):
        self.eval_rewards = []
        for _ in range(self.eval_steps):
            self.eval_rewards.append(self._eval_step())
        return self.eval_rewards

    def learn(self,handler,learn_limit=int(1e6),poison_queue=None,sm_pars={'lambda':.1, 'iters':10}):
        # Making model set.
        self.modset.make_models(handler,sm_pars=sm_pars)
        self.env = eme.FakeWorm(self.modset)

        # Learning loop. 
        learn_eps = 0
        no_poison = True
        ep_lim = learn_limit//self.num_steps

        while learn_eps<ep_lim and no_poison:
            for i in range(self.num_steps):
                self.rewards.append(self._learn_step())
            learn_eps+=1
            if poison_queue is not None:
                no_poison = poison_queue.empty()
        self.env.reset()
        return self.agent.Qtab
        
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

    def __init__(self,agent,worm,act_spacing=3):
        # Start a worm (worm is the env ProcessedWorm)
        # agent should be an agent class with averaged qtables from a multiprocessed run. 
        self.worm = worm
        self.agent = agent
        self.traj = {}
        self.eval_traj = {}
        self.act_spacing = act_spacing

    def eval_ep(self,cam,task,fname,eval_eps=1):
        # Runs an evaluation episode on the worm and returns the total rewards collected.
        self.eval_traj = {}
        obs = self.worm.realobs2obs(self.worm.reset(cam,task))
        done = False
        st,ep = 0,0
        rews = []
        while ep < eval_eps:
            if st%self.act_spacing==0:
                action = self.agent.eval_act(obs)
            next_obs, rew, done, info = self.worm.step(action, cam, task, sleep_time=0) 
            rews.append(rew)
            if next_obs is False:
                return False
            ut.add_to_traj(self.eval_traj, info)
            obs = self.worm.realobs2obs(next_obs)
            st+=1
            task.write(0)
            if done: 
                ep+=1
                if ep<eval_eps:
                    obs = self.worm.realobs2obs(self.worm.reset(cam,task))
            
        with open(fname,'wb') as f:
            pickle.dump(self.eval_traj, f)
        task.write(0)
        return rews

    def eval_ep_fake(self,fname,eval_eps=1):
        # Runs an evaluation episode on the worm and returns the total rewards collected.
        self.eval_traj = {}
        obs = self.worm.realobs2obs(self.worm.reset())
        done = False
        st,ep = 0,0
        rews = []
        while ep < eval_eps:
            if st%self.act_spacing==0:
                action = self.agent.eval_act(obs)
            next_obs, rew, done, info = self.worm.step(action,sleep_time=0) 
            rews.append(rew)
            if next_obs is False:
                return False
            ut.add_to_traj(self.eval_traj, info)
            obs = self.worm.realobs2obs(next_obs)
            st+=1
            if done: 
                ep+=1
                if ep<eval_eps:
                    obs = self.worm.realobs2obs(self.worm.reset())
            
        with open(fname,'wb') as f:
            pickle.dump(self.eval_traj, f)
        return rews

    def full_run(self, num_eps, fname, eps_vector=None,eval_len=300,poison_queue=None):
        # Runs a number of episodes of runtype. Saves trajectory at end.
        # eps_vector is a list of epsilon values for each episode. If there's one value [eps], then 
        # every episode uses the same value.
        #
        # Saves traj 
        if eps_vector is None:
            eps_vector = np.zeros(num_eps)+.1
        elif len(eps_vector)==1:
            eps_vector = np.zeros(num_eps)+eps_vector

        cam,task = init_instruments()
        time.sleep(1) # For some reason we need this 
        check = self.eval_ep(cam,task,fname[:-4]+'_eval_start.pkl',steps=eval_len)
        task.write(0)
        self.traj = {} 
        for ep in range(num_eps):
            check = self.eps_greedy_ep(cam,task,eps_vector[ep]) # Background is reset each time this is called
            task.write(0)
            if check is False:
                if poison_queue is not None:
                    poison_queue.put('STOP')
                raise Exception('HT check failed.')
                
        self.save_traj(fname) 
        task.close()
        cam.exit()
        if poison_queue is not None:
            poison_queue.put('STOP')

    def eps_greedy_ep(self,cam,task,epsilon):
        self.agent.epsilon = epsilon
        obs = self.worm.realobs2obs(self.worm.reset(cam,task))
        done = False
        st=0
        while not done:
            if st%self.act_spacing==0:
                action = self.agent.act(obs)
            next_obs, rew, done, info = self.worm.step(action, cam, task, sleep_time=0)
            if next_obs is False:
                return False
            ut.add_to_traj(self.traj, info)
            obs = self.worm.realobs2obs(next_obs) 
            st+=1
        return True
    
    def check_if_tired(self,fname,active_for=.25):
        # Takes a trajectory file and checks if the worm has disappeared or isn't detectable for
        # more than (1-active_for) of the time.
        with open(fname,'rb') as f:
            tired_traj = pickle.load(f)
        tired_for = tired_traj['reward'].count(0)/len(tired_traj['reward'])
        if tired_for>(1-active_for):
            raise Exception('Worm is tired now.')

    def save_traj(self,fname):
        with open(fname,'wb') as f:
            pickle.dump(self.traj, f)



def get_init_traj(fname, worm, episodes, act_rate=3, rand_probs=[.5,.5]):
    # act_rate is minimum amt of time passed before actions can change.
    cam,task = init_instruments()
    trajs = {}

    for i in range(episodes):
        worm.reset(cam,task)
        done = False
        t=0
        while done is False: 
            if t%act_rate == 0:
                action = np.random.choice([0,1],p=rand_probs)
            obs, rew, done, info = worm.step(action,cam,task)
            ut.add_to_traj(trajs, info)
            t+=1

    cam.exit()
    task.write(0)
    task.close()
    with open(fname,'wb') as f:
        pickle.dump(trajs,f)

def combine_learners(lea_outs):
    # Just outputs averaged Q tables, so output is shape [144,2]

    # First check that everything was successful
    learned_qs = []
    for lea in lea_outs:
        if not lea.successful():
            raise InputError('At least one of the learners was unsuccessful.')
        learned_qs.append(lea.get())

    output_shape = [144,2]
    averaged = np.zeros(output_shape)
    for qs in learned_qs:
        averaged += (1/len(lea_outs))*qs
    return averaged

def make_learner_list(num_learners, worm_pars={'num_models':1, 'frac':.5}, **agentpars):
    learners = []
    for i in range(num_learners):
        agent = tab.Q_Alpha_Agent(**agentpars)
        learners.append(Learner(agent, 'a'+str(i), worm_pars=worm_pars))
    return learners

