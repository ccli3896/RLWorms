'''
General data collection script
'''

import cv2
import pickle
import os
from pyueye import ueye
from pypyueye import Camera, utils
import nidaqmx

import time
from datetime import datetime
import numpy as np
import argparse

import utils


def main(minutes, cam_id, randomrate=.1, lightrate=3, fps=3):

    # Make folder name to save all image data in, by timestamp
    nowtime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_folder = f'./Data/{nowtime}/'
    os.makedirs(save_folder)

    # Initialize hardware: camera and high-powered LED ('task' object)
    cam,task,cam_params = utils.init_instruments(cam_id)

    # Number of frames to track animal for
    frames = int(minutes*60*args.fps)

    # For cases where I just want the initial picture
    if minutes==0:
        frames = 3 

    light = []

    # Throw out first two images
    for i in range(2):
        img = utils.grab_im(cam,None,cam_params)

    # For episode length, take picture and save it to folder
    for i in range(frames):
        img = utils.grab_im(cam,None,cam_params)
        cv2.imwrite(f'{save_folder}/{i}.jpg',img)

        # If I want random light flashes:
        if task is not None:

            # Every [lightrate] frames, choose to randomly turn light on with probability randomrate
            if i%args.lightrate==0:

                # Turn light on if random number is less than randomrate
                # Write light history to list
                if np.random.rand() < randomrate:
                    task.write(cam_params['light_amp'], auto_start=True)
                    task.stop()
                    [light.append(1) for _ in range(lightrate)]

                # If random number is not less than randomrate, turn off and write light off history to list
                else:
                    task.write(0, auto_start=True)
                    task.stop()
                    [light.append(0) for _ in range(lightrate)]

    # Save light data
    with open(f'{save_folder}/light.pkl','wb') as f:
        pickle.dump(light,f)
    
    # Close hardware
    utils.exit_instruments(cam,task)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Collects random data.')
    parser.add_argument('minutes', type=int, help='Minutes of data to collect.')
    parser.add_argument('cam_id', type=int, help='Camera ID, 1 or 2')
    parser.add_argument('--randomrate', type=float, default=.1, help='Probability of light on. Default .1.')
    parser.add_argument('--lightrate', type=int, default=3, help='Number of frames before a light decision is made. Default 3.')
    parser.add_argument('--fps', type=float, default=3, help='Frames per second. Must be =<3. Default 3.')
    args = parser.parse_args()
    print(args)

    main(args.minutes, args.cam_id, randomrate=args.randomrate, lightrate=args.lightrate, fps=args.fps)