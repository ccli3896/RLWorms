'''
Image processing functions as of 4.27.21.
~25 ms for full set of functions
'''

import numpy as np
import cv2
import os
import time
from utils_tutorial import *

from skimage.morphology import skeletonize
'''
Initial functions: defining mask, background, 
'''

def make_mask(bg,size=50):
    center = tuple(map(int,np.flip(np.array(bg.shape)[:2])/2)) # Makes the center of the circle the center of the img
    radius = int(np.mean(center))-size
    img = np.zeros(bg.shape, dtype="uint8")
    img = cv2.bitwise_not(cv2.circle(img, center, radius, (255,255,255), thickness=-1))
    return img

def define_endpt_kernels():
    # To save time in skeleton_endpoints().
    kernels = [np.array((
            [-1, -1, 0],
            [-1,  1, 0],
            [-1, -1, 0]), dtype="int")]
    [kernels.append(np.rot90(kernels[i])) for i in range(3)]
    not_kerns = [np.array((
            [0,0,1],
            [0,1,0],
            [0,0,1]), dtype='int')]
    [not_kerns.append(np.rot90(not_kerns[i])) for i in range(3)]
    return kernels, not_kerns

def make_bg(cam,t):
    bg = grab_im(cam,None).astype('float64')
    start_time = time.monotonic()
    i=0
    while time.monotonic()-start_time < t:
        bg -= 1/(i+1)*(cv2.subtract(bg,grab_im(cam,None).astype('float64')))
        i+=1
    bg = bg.astype('uint8')[:,:,0]
    return bg

def load_templates():
    DEG_INCR = 30
    templates = []
    for i in np.arange(0,360,DEG_INCR):
        templates.append(np.array(cv2.imread('./Templates/template'+str(i)+'.jpg')[:,:,0]))
    return templates

'''
Taking big picture and extracting worm
'''

def find_worm(img,mask,bg,threshold=8,buffer=1,imsz=64,pix_num=15):
    # 2.5 ms
    # Im right from camera. Returns worm image (unthresholded) and location of center of contour.
    worm_im = np.zeros((imsz,imsz),dtype='uint8')
    im = cv2.subtract(cv2.subtract(img,mask),bg)
    retval, th = cv2.threshold(im, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY_INV)
    a,b = cv2.findContours(th,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find object with the biggest bounding box that's not the whole plate
    mx = (0,0,0,0) 
    mx_area = 0
    for cont in a:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area and area < (imsz*4)**2: # imsz**2 is too big for a worm!
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx
    center = np.array([y+h/2,x+w/2],dtype=int)
    if mx_area > pix_num: # and ((w<imsz) & (h<imsz)):
        worm = img[y-buffer:y+buffer+h, x-buffer:x+buffer+w][:imsz,:imsz]
        #percs = np.sort(worm.flatten())
        #worm_im = cv2.add(worm_im,int(percs[pix_num]))
        # worm_im[(imsz-h)//2-buffer:(imsz-h)//2+h+buffer,
        #     (imsz-w)//2-buffer:(imsz-w)//2+w+buffer] = worm
        worm_im = worm
        return worm_im, center
    else:
        return None, None

def get_wormies(img,process=True):
    # Return skeleton and threshold image.
    # 80 us per loop
    # If process is true, remove any small spots by checking contours. 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    blurred = cv2.GaussianBlur(img,(3,3),0)
    th2 = cv2.adaptiveThreshold(cv2.bitwise_not(blurred),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                 cv2.THRESH_BINARY,15,2) #15,2 for dim lighting
    medblur = cv2.medianBlur(th2,3)
    #medblur = cv2.morphologyEx(medblur, cv2.MORPH_CLOSE, kernel)

    if process:
        cont,b = cv2.findContours(medblur,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        max_c = 0
        for c in cont:
            if len(c) > max_c:
                max_c = len(c)
                mask = np.ones(img.shape[:2], dtype="uint8") * 255
                cv2.drawContours(mask,[c],-1,0,-1)
        medblur = cv2.bitwise_or(medblur,mask)

    skeleton = skeletonize(~medblur//255)
    skeleton = skeleton.astype(np.uint8)*255

    return medblur, skeleton

'''
Get endpoints. Need skeleton.
'''

def skeleton_endpoints(skel, kernels, not_kerns):
    # Finds endpoints of skeleton.
    # ~150 us per skeleton
    skel = cv2.copyMakeBorder(skel,3,3,3,3,cv2.BORDER_CONSTANT,value=0)
    endpts = []
    for i,kernel in enumerate(kernels):
        inds = np.array(np.where(cv2.morphologyEx(skel, cv2.MORPH_HITMISS, kernel)),dtype=int)
        if inds.size > 0:
            inds = inds.reshape(2,-1)
            for j in range(inds.shape[1]):
                # Check if different from not_kernel.
                # If different, then can include as endpoint.
                if np.bitwise_xor(skel[inds[0,j]-1:inds[0,j]+2,inds[1,j]-1:inds[1,j]+2]//255,
                                  not_kerns[i]).any():
                    endpts.append(tuple(inds[:,j]))
    return np.array(list(dict.fromkeys(endpts)))-3 # To remove duplicates

'''
For head and body angles.
Need skel for body angle. Need threshold, endpoints, templates for HT angles.
'''

def get_body_angle(skel,discretization=30):
    vy,vx,y,x = cv2.fitLine(np.vstack(skel.nonzero()).T, cv2.DIST_L2,0,0.01,0.01)
    body_angle = np.round((np.arctan2(-vy,vx)*180/np.pi)/discretization)*discretization
    if body_angle==180: body_angle=-180
    return body_angle


def get_HT_ims(th,endpoints,padby=5):
    # 4 us
    # Assumes template size of 7x7
    def get_one_HT_im(th,endpt):
        tsz = 9
        return th[endpt[0]-tsz//2:endpt[0]+tsz//2+1,endpt[1]-tsz//2:endpt[1]+tsz//2+1]

    endpts = endpoints.copy()
    th_pad = cv2.copyMakeBorder(th, padby,padby,padby,padby, None, value=255)
    endpts += padby
    if len(endpts)==2:
        return get_one_HT_im(th_pad,endpts[0,:]), get_one_HT_im(th_pad,endpts[1,:])
    else:
        return [get_one_HT_im(th_pad,endpts[0,:])]

def get_HT_angles(HTs,templates):
    # 350 us
    DEG_INCR = 30
    angs = np.zeros(len(HTs))
    for j,h in enumerate(HTs):
        scores = []
        for i,t in enumerate(templates):
            res = cv2.matchTemplate(cv2.bitwise_not(h),t,cv2.TM_SQDIFF)
            scores.append(np.min(res))
        angs[j] = np.argmin(scores)*DEG_INCR
    angs[angs>180]-=360
    return angs.astype(int)
