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

# UTILITIES ##########################################################
# UTILITIES ##########################################################
# UTILITIES ##########################################################

def load_templates():
    templates = [[],[]]
    for i in np.arange(0,360,DEG_INCR):
        templates[0].append(cv2.imread('Templates/template'+str(i)+'.jpg')[:,:,0])  
    for i in np.arange(0,180,DEG_INCR):
        templates[1].append(cv2.imread('Templates/body'+str(i)+'.jpg')[:,:,0])
    return templates

def pt_dist(pt1,pt2):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return np.sqrt(np.sum(np.square(pt1-pt2)))

def proj(x,y):
    # Projection of x onto y
    if np.linalg.norm(y)==0:
        return 0
    else:
        return np.dot(x,y)/np.linalg.norm(y)

def relative_angle(alpha,beta):
    # Returns angle of alpha relative to beta (positive means alpha is CCW)
    # Range is [-180,180)
    theta = alpha - beta
    if theta >= 180:
        theta -= 360
    elif theta < -180:
        theta += 360
    return theta

def init_instruments(pixelclock=90):
    # Initializes all instruments
    task = nidaqmx.Task()
    task.ao_channels.add_ao_voltage_chan("Dev1/ao0")

    cam = Camera(device_id=0,buffer_count=3)
    cam.init()
    cam.set_colormode(ueye.IS_CM_MONO8)
    cam.set_aoi(200,0,2160,1920) # Full range is wxh = (2560,1920)
    cam.set_pixelclock(pixelclock) # Needs USB 3
    cam.set_fps(20) # It goes to max allowed
    cam.set_exposure(10) # Arbitrary, but 20 is probably too high

    return cam, task

def grab_im(cam,bg):
    # Getting buffer
    buff = utils.ImageBuffer()

    sleep_time = (1/cam.get_fps())*2
    # Allocates mem
    utils.check(ueye.is_AllocImageMem(cam.h_cam,
                          2160, 1920, 8,
                          buff.mem_ptr, buff.mem_id))
    # Set active
    utils.check(ueye.is_SetImageMem(cam.h_cam, buff.mem_ptr, buff.mem_id))
    # Get image
    utils.check(ueye.is_FreezeVideo(cam.h_cam,ueye.IS_DONT_WAIT))
    time.sleep(sleep_time)
    imdata = utils.ImageData(cam.handle(), buff).as_1d_image()
    # Free memory
    ueye.is_UnlockSeqBuf(cam.h_cam,ueye.IS_IGNORE_PARAMETER,buff.mem_ptr)
    utils.check(ueye.is_FreeImageMem(cam.h_cam, buff.mem_ptr, buff.mem_id))
    if bg is None:
        return imdata
    else:
        return cv2.subtract(imdata,bg)

def make_vec_bg(cam,task,light_vec,total_time=60,res=1,mask_it=True):
    # Acquire background by running for a while and averaging.
    # Adds a circle mask for plate if mask_it is True. 
    # Takes vector of light magnitudes
    # total_time is in seconds; res is in seconds. Must be more than flash
    
    def get_bg(cam,total_time,res,light=0,flash=.2):
        # The base function for getting a background image
        task.write(light)
        time.sleep(flash/2)
        bg = grab_im(cam,None).astype('float32')
        time.sleep(flash/2)
        task.write(0)
        
        for i in range(int(total_time//res)):
            time.sleep(res-flash)
            print(f'{i} sec \r',end='')
            task.write(light)
            time.sleep(flash/2)
            bg = bg - 1/(i+1)*(bg-grab_im(cam,None).astype('float32'))
            time.sleep(flash/2)
            task.write(0)

        return bg.astype('uint8')

    def make_mask(bg):
        center = tuple(map(int,np.flip(np.array(bg.shape)[:2])/2)) # Makes the center of the circle the center of the img
        radius = int(np.mean(center))-25
        img = np.zeros(bg.shape, dtype="uint8")
        img = cv2.bitwise_not(cv2.circle(img, center, radius, (255,255,255), thickness=-1))
        return img
    
    bgs = []
    for light in light_vec:
        bgs.append(get_bg(cam,total_time,res,light=light))
    task.write(0)
    if mask_it:
        mask = make_mask(bgs[0])
        bgs = [cv2.add(mask,bg) for bg in bgs]
    return bgs

class Timer:
    # Is a timer that independently keeps track of times and can check whether it's above some fixed interval.
    def __init__(self,interval):
        self.chkpt = time.monotonic() # start time
        self.t = 0
        self.interval = interval
        self.pausevar = 0
    def update(self):
        self.t = time.monotonic() - self.chkpt
    def check(self):
        if self.t > self.interval:
            self.chkpt = time.monotonic()
            self.t = 0
            return True
        else:
            return False
    def reset(self):
        self.chkpt = time.monotonic()
        self.t = 0
    def pause(self):
        self.pausevar = time.monotonic()
    def unpause(self):
        self.pausevar = self.pausevar - time.monotonic() # Sees how much time has elapsed since pause
        self.chkpt = self.chkpt + self.pausevar 

# WORM FUNCTIONS ##########################################################
# WORM FUNCTIONS ##########################################################
# WORM FUNCTIONS ##########################################################

def find_worm_boxes(im,buffer=30,brightness=2000):
    # Finds worms by finding peaks in sums over dimensions. 
    # Checks all intersections for possible worms by setting a brightness threshold.
    # Note that one worm is about 2000 units 
    # Sorted by loc on x axis
    
    def find_peaks(sum_vec,max_worms=1):
        # Finds local maxima that are high enough to probably be a worm/bright spot
        mu = np.mean(sum_vec)
        sig = np.std(sum_vec)
    
        peaks = []
        num_worms = 0
        while np.max(sum_vec) > mu+sig and num_worms<max_worms:
            peaks.append(np.argmax(sum_vec))
            low_ind = np.max([0,peaks[-1]-buffer])
            high_ind = np.min([peaks[-1]+buffer,len(sum_vec)])
            sum_vec[low_ind:high_ind] = np.zeros(high_ind-low_ind)
            num_worms+=1
        return peaks
        
    
    if len(im.shape)>2:
        im = im[:,:,0]
    threshold = 20
    ret,thresh = cv2.threshold(im,threshold,255,0)
    
    # First finds proposal spots
    peaks_x = find_peaks(np.sum(thresh,axis=0))
    peaks_y = find_peaks(np.sum(thresh,axis=1))
    p_worms = []
    t_worms = []
    for i in peaks_x:
        for j in peaks_y:
            p_worms.append(im[int(j-buffer):int(j+buffer+1),int(i-buffer):int(i+buffer+1)])
            # Makes worm box centered on peak brightness with [buffer] pixels on either side.
            t_worms.append(thresh[int(j-buffer):int(j+buffer+1),int(i-buffer):int(i+buffer+1)])
            # Makes thresholded worm box for loc
    
    # Checks proposal boxes and throws out ones that probably don't have worms
    worms = []
    centers = []
    peak_centers = []
    for i,worm in enumerate(p_worms):
        if sum(worm.flatten())>brightness: 
            # Saves centers as COM in found box
            sumx,sumy = np.sum(t_worms[i],axis=0),np.sum(t_worms[i],axis=1)
            im_sz = buffer*2+1
            centers.append(np.array([np.sum(np.arange(im_sz)*sumx) / np.sum(sumx), np.sum(np.arange(im_sz)*sumy) / np.sum(sumy)]))
            peak_centers.append(np.array([peaks_x[i//len(peaks_y)],peaks_y[i%len(peaks_y)]]))
            centers[-1] = centers[-1]+peak_centers[-1]-buffer
            worms.append(worm)
    # Sort by loc on x axis
    if len(centers)>1:
        c_ind = np.argsort(np.array(centers)[:,0]).tolist()
        centers = [x for _,x in sorted(zip(c_ind,centers))]
        peak_centers = [x for _,x in sorted(zip(c_ind,peak_centers))]
        worms = [x for _,x in sorted(zip(c_ind,worms))]
    return centers,worms,peak_centers

def find_angs(img,loc,ref_pt,templates,buffer=30,th_ind=-80):
    # Returns best_matches, which has variables about best template matches. 
    # [match score]
    # [   angle   ]
    # [ x0   x1   ]
    # [ y0   y1   ]
    # ref_pt is to sort the two endpoints. Lists the one closest to ref_pt first.
    # Note ref_pt should be relative to whole image.
    # loc is where the center of img is, in the whole frame.
    # buffer should match buffer variable in find_worm_boxes.
    # th_ind is the index of the threshold to set. A reasonable one for 5MP cameras is the 55th brightest pixel.
    
    # SUBFUNCTIONS **************************************************
    def same_point(best_matches):
        # Takes best_matches array [scores;deg;locs, n]
        # and returns same_flag if two points have been counted twice and best_matches with the worse of the two
        # moved to the end
        _,n = best_matches.shape
        for i in range(n):
            for j in np.arange(n-i-1)+1+i:
                if pt_dist(best_matches[2:,i],best_matches[2:,j]) < 4:
                    worse = [i,j][np.argmax(best_matches[0,[i,j]])] # returns the i or j that's worse
                    best_matches[:,worse] = np.array([1e10,-1,-5,-5])
                    return best_matches[:,np.argsort(best_matches[0,:])]
        return best_matches       
    
    # MAIN FUNCTION ************************************************
    img = img.astype('uint8')
    threshold = np.sort(img.flatten())[th_ind]
    ret, img = cv2.threshold(img,threshold,255,0)
    best_matches = np.vstack((np.array([[1e10,1e10,1e10], [-1,-1,-1]]), np.zeros((2,3))))

    for i,template in enumerate(templates):
            
        res = cv2.matchTemplate(img,template,cv2.TM_SQDIFF)
        deg = i*DEG_INCR
      
        min_ind = np.unravel_index(res.argmin(), res.shape)
        best_matches[0,2] = res[min_ind]
        best_matches[1,2] = deg
        best_matches[2:,2] = np.flip(min_ind)+3.5-buffer+loc # +3.5 is for template size (7,7)

        sort_best = np.argsort(best_matches[0,:])
        best_matches = best_matches[:,sort_best]
        
        # Make sure points aren't identical
        if i>2:
            best_matches = same_point(best_matches)        

    best_matches = best_matches[:,:2]
    dist_sort = np.argsort([pt_dist(ref_pt,best_matches[2:,0]),pt_dist(ref_pt,best_matches[2:,1])])

    best_matches = best_matches[:,dist_sort]

    best_matches = np.around(best_matches,1).astype(int)
    return best_matches

def find_body(worm,bodies):
    # Finds body orientation in degrees. Uses endpoints to figure out direction after template matching.
    def get_vec(deg):
        return [np.cos(deg),np.sin(deg)]

    res = [min(cv2.matchTemplate(worm['img'],body,cv2.TM_SQDIFF).flatten()) for body in bodies]
    deg = np.argmin(res) * DEG_INCR *pi/180 # to radians
    end_ang = np.arctan2(-(worm['endpts'][1,0] - worm['endpts'][1,1]), worm['endpts'][0,0] - worm['endpts'][0,1])
                            # y1 - y0, x1 - x0
    end_vec = get_vec(end_ang)
    deg_vec0 = get_vec(deg)
    deg_vec1 = get_vec(deg+pi)

    # Returns 0 if original template is best. Returns 1 if need to add 180 deg
    better_deg = np.argmax([proj(end_vec,deg_vec0),proj(end_vec,deg_vec1)])
    return round(deg*180/pi + 180*better_deg)

def find_worms(im, templates, bodies, ref_pts=None,num_worms=1,brightness=2000):
    # Returns worms, list of dictionaries with worm traits.
    # threshold is 20 for lowmag, 30 for highmag.
    # templates is a list in ascending order of all worm templates (assuming increments DEG_INCR deg)
    # ref_pts is a list of reference points for the probable head of the worm

  # Finds all worms in one image and returns a worms list of dictionaries.
  # In the dictionary, 
    # img is the input image, cropped to worm, black background.
    # loc is xy coordinates of worm brightness center
    # scores is template score pair
    # angs is best angles
    # endpts is coords of endpoints, [x0,x1]
    #                                [y0,y1]

    if ref_pts is None:
        ref_pts = [[0,0] for _ in range(num_worms)]
        
    # Get probable centers and worm images, not thresholded
    centers,boxes,box_centers = find_worm_boxes(im,brightness=brightness)
    if len(boxes) == 0:
        #print('No worms found')
        return None
    
    worms = [{} for i in range(num_worms)]
    
    for wrm in range(num_worms):
    # for each boxed worm get labels

        # Cropping image for each box and inits
        #ret, worm = cv2.threshold(boxes[wrm],threshold,255,0)
        worms[wrm]['loc'] = centers[wrm]
        worms[wrm]['img'] = boxes[wrm]

        # Get best angles by template matching endpoints
        best_matches = find_angs(boxes[wrm],box_centers[wrm],ref_pts[wrm],templates)
        worms[wrm]['scores'] = best_matches[0,:]
        worms[wrm]['angs'] = best_matches[1,:]
        worms[wrm]['endpts'] = best_matches[2:,:]
        worms[wrm]['body'] = find_body(worms[wrm],bodies)
    return worms

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
    
    return head, SWITCH

def find_worm_only(im, brightness=2000):
    # Returns worms with location. Right now suitable for one worm.

    # Finds all worms in one image and returns a worms list of dictionaries.
    # In the dictionary, 
        # img is the input image, cropped to worm, black background.
        # loc is xy coordinates of worm brightness center
        
    # Get probable centers and worm images, not thresholded
    centers,boxes,box_centers = find_worm_boxes(im,brightness=brightness)
    if len(boxes) == 0:
        print('No worms found')
        return None
    
    worms = [{} for i in range(num_worms)]
    
    for wrm in range(num_worms):
        # Cropping image for each box and inits
        worms[wrm]['loc'] = centers[wrm]
        worms[wrm]['img'] = boxes[wrm]

    return worms

# PLOT FUNCTIONS ##########################################################
# PLOT FUNCTIONS ##########################################################
# PLOT FUNCTIONS ##########################################################

def plot_track(track):
    fig,ax = plt.subplots(1)
    fig.set_size_inches((8,8))
    NPOINTS = track.shape[0]
    ax.set_prop_cycle('color',plt.cm.winter(np.linspace(0,1,NPOINTS)))
    for i in range(NPOINTS-1):
        ax.scatter(track[i,1],-track[i,2])
        ax.set_aspect('equal','box')
    return fig,ax

# WHOLE FUNCTIONS ##########################################################
# WHOLE FUNCTIONS ##########################################################
# WHOLE FUNCTIONS ##########################################################

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
                
                #if SWITCH:
                    #print('\t\tSwitched')
            START = False
            
        # Update all timers
        elapsed.update()
        track_el.update()
        ht_el.update()
        
    task.write(0)
    return np.array(track),np.array(angs),np.array(lights)