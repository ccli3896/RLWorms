from improc import *
from worm_env import *
import random
import pickle
from datetime import datetime

env = ProcessedWorm(0,ep_len=500)

episodes=5
act_rate = 5 # actions switch at the fewest after this many frames
                # assuming a framerate of about 5
trajs = {}

for i in range(episodes):
    env.reset()
    done = False
    print(i)
    t=0
    while done is False:
        if t%act_rate == 0:
            action = random.choice([0,1])
        obs, rew, done, info = env.step(action)
        add_to_traj(trajs, info)
        t+=1

env.close()
fname = './Data/traj'+datetime.now().strftime('%d-%m-%Y_%H-%M-%S')+'.pkl'
with open(fname,'wb') as f:
    pickle.dump(trajs,f)