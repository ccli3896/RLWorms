'''
Some utilities that I'm not sure where to put otherwise
'''

import numpy as np
from pyueye import ueye
from pypyueye import Camera, utils
import nidaqmx
import time 
import cv2 


'''
General quick functions
'''

def worm_bound(a):
    # Make sure angles always stay within +/-180 degrees
    if hasattr(a,'__len__'):
        a = np.array(a)
        a = np.where(a<-180, a+360, a)
        a = np.where(a>=180, a-360, a)
        return a

    else:
        if a<-180: return a+360
        elif a>=180: return a-360
        else: return a

def ang_between(a,b):
	# Both angles must be worm-bounded in [-180,180)
    diff = np.abs(a-b)
    if diff > 180:
        diff = np.abs(diff-360)
    return diff

def projection(body_angle, mvt_angle, speed):
	# Gets speed in direction of the body angle.
	ang = ang_between(body_angle,mvt_angle)
	return speed*np.cos(ang*np.pi/180)


'''
Image acquisition and processing
'''

def off(cam_id):
    # Reset
    cam,task,cam_params = init_instruments(cam_id)
    exit_instruments(cam,task)

def init_instruments(cam_id):
    # Initializes instruments for the two different rigs. Different camera models and resolutions.
    # These are the parameters that seem to work and allow me to use the same image processing settings, mostly

    if cam_id==1:
        cam_params = {
            'cam_id': 1,
            'im_width': 2160,
            'im_height': 1920,
            'pixelclock': 90,
            'exposure': 12,
            'fps': 10,
            'wait_time': .29,
            'light_amp': 2, # For power of roughly 5 mA
            'task_id': '1',
            'threshold': 8,
        }

    elif cam_id==2:
        cam_params = {
            'cam_id': 2,
            'im_width': 1280,
            'im_height': 1024,
            'pixelclock': 20,
            'exposure': 1.25,
            'fps': 9,
            'wait_time': .33,
            'light_amp': 1, # Roughly 5 mA (different light)
            'task_id': '2',
            'threshold': 10, # 25 with green light, 18 with blue (?)
        }

    task = nidaqmx.Task()
    task.ao_channels.add_ao_voltage_chan(f"Dev{cam_params['task_id']}/ao0")
    task.start()

    cam = Camera(device_id=cam_params['cam_id'],buffer_count=3)
    cam.init()
    cam.set_colormode(ueye.IS_CM_MONO8)
    cam.set_aoi(200,0,cam_params['im_width'],cam_params['im_height']) # Full range is w x h = (2560,1920)
    cam.set_pixelclock(cam_params['pixelclock']) # Needs USB 3
    cam.set_fps(cam_params['fps']) # It goes to max allowed
    cam.set_exposure(cam_params['exposure']) # Arbitrary, but 20 is probably too high

    return cam, task, cam_params


def grab_im(cam,bg,cam_params):
    # Getting buffer
    buff = utils.ImageBuffer()

    sleep_time = cam_params['wait_time']
    # Allocates mem
    utils.check(ueye.is_AllocImageMem(cam.h_cam,
                          cam_params['im_width'], cam_params['im_height'], 8,
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

    # Resize image if it's the lower res camera
    if cam_params['cam_id']==2:
        imdata = cv2.resize(imdata, (2160,1728), interpolation=cv2.INTER_AREA)

    if bg is None:
        return imdata
    else:
        return cv2.subtract(imdata,bg)

def make_bg(cam, cam_params, bgtime=30):
    # Make the background to be subtracted from images later, for worm-finding
    start = time.monotonic()
    bg = grab_im(cam,None,cam_params).astype('float64')
    i=1
    print('Making bg')
    while time.monotonic()-start < bgtime:
        bg -= 1/(i+1)*(cv2.subtract(bg,grab_im(cam, None, cam_params).astype('float64')))
        i+=1
    print('Done with bg')
    return bg.astype('uint8')


def exit_instruments(cam,task):
    task.write(0)
    task.close()
    cam.exit()
