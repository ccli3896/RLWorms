'''The sample dataset is from an evaluation episode with trained agents on line 301. 
Images are stored in `./Tutorials/Worm image data/imgs/`. 
Track data is saved in a `.pkl` file and two JPG images: one colored by time and the other colored by agent actions, 
read from `./Tutorials/Worm image data/actions2021-11-02_13-40-29.pkl`. 

Raw image datasets are very large so have not been uploaded in their entirety. They are available upon request. 
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import argparse
import os


import SAC as sac
from utils_tutorial import *
import improc_v as iv
import worm as w


def main():
    # Number of images to track worm for
    n_ims = 500
    target = [1080,-1720] # Down target

    # Initialize worm object for feature extraction
    fw = w.FakeWorm(f'./Tutorials/Worm image data/imgs/',cam_id=1)

    # Processing images in folder and getting locations
    x,y = [],[]
    body,head = [],[]

    # If it's time to make a new background, make a new background by averaging a set of images
    fw.bg = fw.make_bg(start=0,end=n_ims,res=10)

    # Calculates locations and angles at that image; appends to tracked variable lists
    for i in np.arange(n_ims):
        fw.step()
        x.append(fw.locx)
        y.append(fw.locy)
        body.append(fw.body_ang)
        head.append(fw.head_ang)


    # Plotting tracks colored by time and saving
    plt.figure(figsize=(16,16))
    cm = np.arange(len(x))

    # Plot background image
    plt.imshow(fw.bg,cmap='bone')

    # Plot track
    im = plt.scatter(x,-np.array(y),c=cm,marker='.')

    # Plot target
    plt.scatter(int(target[0]),-int(target[1]),c='r')

    # Plot starting point
    plt.scatter(x[0],-y[0])

    # Create a colorbar
    cbar = plt.colorbar(im, ticks=np.arange(0, 500, 60))
    cbar.set_label('Time (s)')
    cbar.ax.set_yticklabels(np.arange(0,500,60)//3)

    # Title
    plt.title('Track colored by time')

    # Save figure
    plt.savefig(f'./Tutorials/Track_coloredbytime.jpg')
    print('Time plot saved')


    # Save track data to dict
    data = {'x':x, 'y':y, 'body':body, 'head':head}
    with open(f'./Tutorials/sampletrack.pkl','wb') as f:
        pickle.dump(data,f)
    

    # Make track colored by light activation and save

    # Open agent action data (binary on/off)
    with open('./Tutorials/Worm image data/actions2021-11-02_13-40-29.pkl', 'rb') as f:
        light = pickle.load(f)

    # Make figure
    plt.figure(figsize=(16,16))

    # Plot background image
    plt.imshow(fw.bg,cmap='bone')

    # Plot track colored by agent action
    plt.scatter(x,-np.array(y),c=light[:n_ims],marker='.')

    # Plot target
    plt.scatter(int(target[0]),-int(target[1]),c='r')

    # Plot starting point
    plt.scatter(x[0],-y[0])

    # Create custom legend patches
    legend_patches = [mpatches.Patch(color='purple', label='Off'),
                      mpatches.Patch(color='yellow', label='On')]

    # Add the legend to the figure
    plt.legend(handles=legend_patches, loc='upper right')

    # Title
    plt.title('Track colored by light on/off')

    # Save
    plt.savefig(f'./Tutorials/Track_coloredbyaction.jpg')
    print('Action plot saved')

if __name__=='__main__':
    main()