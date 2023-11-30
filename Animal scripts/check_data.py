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
    # Parse all arguments
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
    args = parse_args() 
    fw = w.FakeWorm(f'{args.folder}',cam_id=1)

    # Processing images in folder and getting locations
    x,y = [],[]
    body,head = [],[]
    for i in np.arange(args.num_ims):
        #print(i)
        if i%args.bgres==0:
            fw.bg = fw.make_bg(start=i,end=min([args.num_ims,i+args.bgres]),res=10)
        fw.step()
        x.append(fw.locx)
        y.append(fw.locy)
        body.append(fw.body_ang)
        head.append(fw.head_ang)

    # Plotting tracks and saving
    plt.figure(figsize=(16,16))
    cm = np.arange(len(x))
    plt.imshow(fw.bg,cmap='bone')
    plt.scatter(x,-np.array(y),c=cm,marker='.')
    target = args.target.split(',')
    plt.scatter(int(target[0]),-int(target[1]),c='r')
    plt.scatter(x[0],-y[0])

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