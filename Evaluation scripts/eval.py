import numpy as np
import pickle
import argparse
from datetime import datetime
import os

import torch
from torch import nn

import SAC as sac
from utils import *
import improc_v as iv
import worm as w

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    # Parse all arguments
    parser = argparse.ArgumentParser()
    # Experiment variables
    parser.add_argument('cam_id', type=int, help='1 or 2')
    parser.add_argument('line', type=int, help='281, 301, or 352')
    parser.add_argument('target', type=str, help='comma-separated, two integers in pixels from center of plate')
    parser.add_argument('--eptime', type=int, default=600, help='Recording time in seconds (default 600)')
    parser.add_argument('--fps', type=int, default=3, help='Recording frames per second (default 3)')
    parser.add_argument('--bgtime', type=int, default=20, help='Seconds to collect background (default 20)')
    parser.add_argument('--printfreq', type=int, default=5, help='Frames between printing updates (default 5)')
    parser.add_argument('--light_amp', type=int, default=3, help='Light amplitude (default 3)')
    # Agent variables
    parser.add_argument('--seeds', type=int, default=20, help='Number of agents in the ensemble (default 20)')
    parser.add_argument('--framesperstate', type=int, default=15, help='Frames per state (default 15)')
    parser.add_argument('--actions', type=int, default=2, help='Number of discrete actions (default 2 [on/off])')
    parser.add_argument('--nrns', type=int, default=64, help='Neurons in each agent network (default 64)')
    parser.add_argument('--translated', type=int, default=1350, help='Amount to normalize worm distance by. Default 1200.')

    args = parser.parse_args()

    return args

def get_state(worm, translated=900, target=[0,0]):
    # Returns state format: sin(body),cos(body),sin(head),cos(head),locx,locy
    body = worm.body_ang/180*np.pi
    head = worm.head_ang/180*np.pi
    
    x,y = worm.locx-target[0], worm.locy-target[1]
    locx = x/translated
    locy = y/translated
    return [np.sin(body),np.cos(body),np.sin(head),np.cos(head),locx,locy]


def main():
    args = parse_args() 

    # Prepare: ensemble, instruments, folder to save in.
    if args.line==281:
        prelabel = './Actors/sac_actor_L281_NoReg_t15_tr900_n64'
        postlabel = 'fin_'
    elif args.line==301:
        prelabel = './Actors/sac_actor_L301_0sample100'
        postlabel = 'fin_'
    elif args.line==352:
        prelabel = './Actors/sac_actor_L352'
        postlabel = 'fin_'
    
    actors = sac.load_actors(prelabel, postlabel, args.seeds,
                            frames=args.framesperstate, actions=args.actions, nrns=args.nrns)
    ens = sac.Ensemble(actors,actions=args.actions)
    cam, task, cam_params = init_instruments(args.cam_id)
    nowtime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_folder = f'./Data/{nowtime}/'
    os.makedirs(save_folder)

    # Init worm object and episode variables, make bg
    worm = w.RealWorm(cam, cam_params, bg_time=args.bgtime)
    frames = args.eptime*args.fps
    target = [int(t) for t in args.target.split(',')]


    # 1. Get an image (save)
    # 2. Get the state 
    # 3. Decide on an action
    # 4. Execute action
    # Loop 1-4, then save state and actions

    # State format: sin(body),cos(body),sin(head),cos(head),locx,locy
    # Initialize state
    current_state = np.zeros((6,args.framesperstate))
    starttime = time.monotonic()
    for fr in range(args.framesperstate):
        worm.step()
        cv2.imwrite(f'{save_folder}pre{fr}.jpg',worm.img)
        current_state[:,fr] = get_state(worm, translated=args.translated, target=target)
    print(time.monotonic()-starttime)

    # Run the episode
    actions = []
    for fr in range(frames):
        # Update state
        worm.step()
        cv2.imwrite(f'{save_folder}{fr}.jpg',worm.img)
        current_state[:,:-1] = current_state[:,1:]
        current_state[:,-1] = get_state(worm, translated=args.translated, target=target)

        # Get action from ensemble
        action = ens(torch.tensor(current_state).view(1,-1).type(torch.float).to(device)).detach().cpu().numpy()
        actions.append(action)
        task.write(action*args.light_amp)

        # Print an update
        if fr%args.printfreq==0:
            print(worm.locx,worm.locy,worm.body_ang,worm.head_ang,action)
            print('\t\t',current_state[:,-1][-2:])
    print(time.monotonic()-starttime)


    # Save actions
    with open(f'{save_folder}actions.pkl','wb') as f:
        pickle.dump(actions,f)
    exit_instruments(cam,task)


if __name__=='__main__':
    main()