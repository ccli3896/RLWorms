import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os


import SAC as sac
from utils import *
import improc_v as iv
import worm as w

def parse_args():
    # Parse all arguments:
    #  - Folder label to read input from
    #  - Number of images to track
    #  - Target coordinates, for plotting
    #  - Rig ID (doesn't make too much of a difference)
    #  - Optional:
    #     - save: whether or not to save track or just see image
    #     - savename: name of saved image
    #     - bgres: how many images to average over to make a background. Can impact tracking a lot; reasonable range 100 to 1800
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='Folder label with /')
    parser.add_argument('num_ims', type=int, help='Number of images in folder')
    parser.add_argument('target', type=str, help='Comma-separated x,y coordinates of target')
    parser.add_argument('cam_id', type=int, help='1 or 2. This is for threshold')
    parser.add_argument('--save', default=True, action=argparse.BooleanOptionalAction, help='Whether to save an image')
    parser.set_defaults(save=True)
    parser.add_argument('--savename', type=str, default='Track', help='Saved image filename. Default Track.')
    parser.add_argument('--bgres', type=int, default=500, help='How often to remake the background. Default 500.')

    args = parser.parse_args()
    print(args)

    return args

def main():
    # Get inputs and initialize worm object for feature extraction
    args = parse_args() 
    fw = w.FakeWorm(f'{args.folder}',cam_id=1)

    # Processing images in folder and getting locations
    x,y = [],[]
    body,head = [],[]

    # Loop over each image
    for i in np.arange(args.num_ims):
        # if it's time to make a new background, make a new background by averaging a set of images
        # res argument in make_bg() is the size of the gap between averaged images: eg if bgres=500 and res=10, the function will average images 0, 10, 20,...,490 at the start
        if i%args.bgres==0:
            fw.bg = fw.make_bg(start=i,end=min([args.num_ims,i+args.bgres]),res=10)

        # Calculates locations and angles at that image; appends to tracked variable lists
        fw.step()
        x.append(fw.locx)
        y.append(fw.locy)
        body.append(fw.body_ang)
        head.append(fw.head_ang)


    # Plotting tracks and saving
    plt.figure(figsize=(16,16))
    cm = np.arange(len(x))
    plt.imshow(fw.bg,cmap='bone')

    # Plot track
    plt.scatter(x,-np.array(y),c=cm,marker='.')
    target = args.target.split(',')

    # Plot the target point
    plt.scatter(int(target[0]),-int(target[1]),c='r')
    
    # Plot the starting point
    plt.scatter(x[0],-y[0])

    # Save figures and tracking data
    if args.save:
        plt.savefig(f'{args.folder}{args.savename}.svg')
        plt.savefig(f'{args.folder}{args.savename}.jpg')
        print('Images saved')
        data = {'x':x, 'y':y, 'body':body, 'head':head}
        with open(f'{args.folder[:-1]}.pkl','wb') as f:
            pickle.dump(data,f)
        
    
    plt.show()

if __name__=='__main__':
    main()