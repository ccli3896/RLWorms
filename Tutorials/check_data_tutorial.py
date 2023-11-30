'''The sample dataset is from an evaluation episode with trained agents on line 301. 
Images are stored in `./Tutorials/Worm image data/imgs/`. 
Track data is saved in a `.pkl` file and two JPG images: one colored by time and the other colored by agent actions, 
read from `./Tutorials/Worm image data/actions2021-11-02_13-40-29.pkl`. 

Raw image datasets are very large so have not been uploaded in their entirety. They are available upon request. 
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os


import SAC as sac
from utils_tutorial import *
import improc_v as iv
import worm as w


def main():
    n_ims = 500
    target = [1080,-1720] # Down target
    fw = w.FakeWorm(f'./Worm image data/imgs/',cam_id=1)

    # Processing images in folder and getting locations
    x,y = [],[]
    body,head = [],[]
    fw.bg = fw.make_bg(start=0,end=n_ims,res=10)

    for i in np.arange(n_ims):
        fw.step()
        x.append(fw.locx)
        y.append(fw.locy)
        body.append(fw.body_ang)
        head.append(fw.head_ang)


    # Plotting tracks colored by time and saving
    plt.figure(figsize=(16,16))
    cm = np.arange(len(x))
    plt.imshow(fw.bg,cmap='bone')
    plt.scatter(x,-np.array(y),c=cm,marker='.')
    plt.scatter(int(target[0]),-int(target[1]),c='r')
    plt.scatter(x[0],-y[0])
    plt.savefig(f'./Track_coloredbytime.jpg')
    print('Time plot saved')


    # Save track data to dict
    data = {'x':x, 'y':y, 'body':body, 'head':head}
    with open(f'sampletrack.pkl','wb') as f:
        pickle.dump(data,f)
    

    # Make track colored by light activation and save
    with open('./Worm image data/actions2021-11-02_13-40-29.pkl', 'rb') as f:
        light = pickle.load(f)
    plt.figure(figsize=(16,16))
    plt.imshow(fw.bg,cmap='bone')
    plt.scatter(x,-np.array(y),c=light[:n_ims],marker='.')
    plt.scatter(int(target[0]),-int(target[1]),c='r')
    plt.scatter(x[0],-y[0])
    plt.savefig(f'./Track_coloredbyaction.jpg')
    print('Action plot saved')

if __name__=='__main__':
    main()