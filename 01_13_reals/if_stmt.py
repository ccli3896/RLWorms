import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os

import nidaqmx # laser output
from pyueye import ueye
from pypyueye import Camera,utils
import time # delay within program
from math import *

DEG_INCR = 30

def if_stmt_angle(direction,cam,task,bgs,templates,bodies,total_time=600,track_res=1,ht_res=3,light_i=1):
    # Makes a worm go in angle direction.
    # Will return the path of the worm in an array where first col is time elapsed, and second and third are
    # x and y coords respectively.
    # ht_res is how many seconds to wait before collecting a new HT.
    
    # Because of the issues in the worm room with reflection when taking pictures with light, this one
    # flashes off for a moment when images are being collected. Thus no with-light backgrounds are necessary.
    
    # total_time is in seconds.

    track = []
    angs = []
    lights = []
    light_vec = [0,light_i]
    light_ind = 0
    START = True
    bg = bgs[0]
    
    # Find initial head endpoint
    print('Finding orientation')
    head,old_loc = find_ht(cam,bg,templates,bodies,runtime=3)

    # Initialize all timers
    elapsed = Timer(total_time)
    track_el = Timer(track_res)
    ht_el = Timer(ht_res)    
    
    while not elapsed.check():
        
        # Collect image and make sure worm exists
        img = grab_im(cam,bg)        
        worms = find_worms(img,templates,bodies,ref_pts=[head],num_worms=1)
        
        if worms is None:
            task.write(0)
            print('Didn\'t find right worm')
            continue
        worm = worms[0]
        head = worm['endpts'][:,0]
        angs.append(worm['angs'])
        
        
        # Deciding light value based on relative angles
        body_dir = relative_angle(worm['body'],direction)
        head_body = relative_angle(worm['angs'][0],worm['body'])
        
        if body_dir*head_body < 0:
            light_ind = 1
        elif body_dir == 0:
            if head_body == 0:
                light_ind = 1
            else:
                light_ind = 0
        else:
            light_ind = 0
        
        # Apply light chosen
        task.write(light_vec[light_ind])
        lights.append(light_vec[light_ind])
            
        # Check: save track and reset head direction
        if track_el.check():
            track.append(np.hstack([elapsed,worm['loc']]))
            print(int(elapsed.t))
            print('\t',round(worm['loc'][0],2),-round(worm['loc'][1],2))
            print('\tbody',worm['body'],'head',worm['angs'][0],'light',lights[-1])
        if ht_el.check():
            if not START:
                head,SWITCH = ht_quick(worm,old_loc)
                old_loc = worm['loc']
                
                if SWITCH:
                    print('\t\tSwitched')
            START = False
            
        # Update all timers
        elapsed.update()
        track_el.update()
        ht_el.update()
        
    task.write(0)
    return np.array(track),np.array(angs),np.array(lights)

def ht_quick(worm,old_loc):
    # Returns probable head endpoint based on movement projections onto angles obtained from find_angs.
    p0 = proj(worm['loc']-old_loc, [np.cos(pi/180*worm['angs'][0]),-np.sin(pi/180*worm['angs'][0])])
    p1 = proj(worm['loc']-old_loc, [np.cos(pi/180*worm['angs'][1]),-np.sin(pi/180*worm['angs'][1])])
    #print(round(p0,2),round(p1,2))
    SWITCH = False
    head = worm['endpts'][:,0]
    if p1 > p0:
        head = worm['endpts'][:,1]
        SWITCH = True
        print('HT switched',head)
    
    return head, SWITCH

def find_ht(cam,bg,templates,bodies,runtime=30):
    # SINCE THIS IS THE ONLY POINT WHERE THE SCRIPT CAN FAIL (I THINK) IT'S REMOVED IN TD SCRIPT

    # Finds the probable head/tail coordinates of worm based on average direction of movement over time.
    # Assumes that center of worm moves more in direction of head than tail.
    # Works by adding up 
    
    # runtime is in seconds
    # bg is the background without light
    
    img = grab_im(cam,bg)
    start_time = time.monotonic()
    worm_i = find_worms(img,templates,bodies,num_worms=1)[0]
        # Initializes first endpoint as the one closest to the origin
    
    travels = [0,0]
    path = []
    
    elapsed=0
    while elapsed < runtime:
        print(f'elapsed {int(elapsed)}\r',end='')
        img = grab_im(cam,bg)
        worm_f = find_worms(img,templates,bodies,ref_pts=[worm_i['endpts'][:,0]],num_worms=1)[0]
        
        # Get movement of center vector, and vectors from center to endpts
        cvec = worm_f['loc']-worm_i['loc'] # vector of mvt of worm center
        if np.linalg.norm(cvec) > 20:
            continue
        endvecs = [worm_f['endpts'][:,0]-worm_f['loc'], worm_f['endpts'][:,1]-worm_f['loc']]
        
        # Add up total distance travelled in direction of each endpoint
        travels[0] = travels[0] + proj(cvec,endvecs[0])
        travels[1] = travels[1] + proj(cvec,endvecs[1])
        
        worm_i = worm_f
        path.append(worm_f['loc'])
        elapsed = time.monotonic() - start_time
    head = worm_f['endpts'][:,np.argmax(travels)]
    old_loc = worm_f['loc']
    print('\n')
    return head,old_loc