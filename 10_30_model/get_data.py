from improc import *
from worm_env import *
import random
import pickle
from datetime import datetime

env = ProcessedWorm(0,ep_len=500)

episodes=5
trajs = {}

for i in range(episodes):
    env.reset()
    done = False
    print(i)
    while done is False:
        obs, rew, done, info = env.step(random.choice([0,1]))
        add_to_traj(trajs, info)

env.close()

fname = './Data/traj'+datetime.now().strftime('%d-%m-%Y_%H-%M-%S')+'.pkl'
with open(fname,'wb') as f:
    pickle.dump(trajs,f)