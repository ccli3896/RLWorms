import improc_v as iv
import cv2
from utils_tutorial import *
import os

class RealWorm():
    '''
    A fake worm class that reads in images and returns descriptors.
    The descriptors saved in the class are:
        locx, locy
        body_ang
        head_, tail_ (endpts)
        head_ang, tail_ang
        curl
        speed
        mvt_ang

    Also of interest:
        track_frames_ (array of last [track] locations)

    '''
    def __init__(self,cam,cam_params,bg_time=30,track=6,HT_sum_len=30,speed_th=.2):
        # Hyperparameters #
        self.track_frames_len_ = track
        self.HT_sum_len = HT_sum_len # Number of points to track to see when to flip
        self.speed_th = speed_th # Minimum speed to count a reversal or flip

        # Init stuff #
        self.cam = cam
        self.bg_time = bg_time
        self.templates = iv.load_templates()
        self.kerns, self.not_kerns = iv.define_endpt_kernels()

        self.cam_params = cam_params
        self.reset()
        self.mask = iv.make_mask(self.bg)
        self.im_num = -1

        

    def __str__(self):
        return f'Worm {self.im_num}: loc {self.locx,self.locy}, endpts {self.head_,self.tail_},\n \
            Body {self.body_ang}, HT {self.head_ang,self.tail_ang},\n \
            Speed {np.round(self.speed,2)}, Mvt angle {np.round(self.mvt_ang,2)}'

    def step(self,im_num=None):

        '''
        Load image and get endpoints, endpoint angles, skeleton, location. 
        Updated attributes:
            locy,locx,body_ang,head_ang,tail_ang,curl,speed,mvt_ang
        '''
        if im_num is None:
            self.im_num += 1
        else:
            self.im_num = im_num
        # Getting worm img and location
        self.img = grab_im(self.cam,None,self.cam_params)
        self.worm, locs = iv.find_worm(self.img,self.mask,self.bg,threshold=self.cam_params['threshold'])
        if self.worm is None:
            print(f'Empty frame {self.im_num}')
            return
        self.locy,self.locx = locs
        self.locy *=-1
        th,skel = iv.get_wormies(self.worm)
        endpts = iv.skeleton_endpoints(skel,self.kerns,self.not_kerns)
        # If there are too many endpoints (bright interfering spots), skip this frame
        if len(endpts)>2:
            print(f'Passed frame {self.im_num}')
            return
        if len(endpts)==0: # Full circle
            self.curl = True
            # Update previous locations
            self.track_frames_[:-1,:] = self.track_frames_[1:,:]
            self.track_frames_[-1,:] = [self.locx, self.locy]
            # Get velocity
            travel_vec = self.track_frames_[-1,:] - self.track_frames_[0,:]
            self.speed = np.linalg.norm(travel_vec) / self.track_frames_len_
            self.mvt_ang = worm_bound(np.arctan2(*np.flip(travel_vec))*(180/np.pi))
            return

        self.body_ang = iv.get_body_angle(skel)
        endpt_angles = iv.get_HT_angles(iv.get_HT_ims(th,endpts),self.templates)
        
        # Fix flip and negative y image issue for endpoints
        endpts = np.fliplr(endpts)
        endpts[:,1] *= -1

        # If it's the first step, initialize some variables.
        # Note: all the flips are for switching x and y variables from image.
        if self.start:
            self.start = False
            self.head_ = endpts[0]
            self.head_ang = endpt_angles[0]

            # If first step, fill in previous locations for speed tracking
            self.track_frames_[:,0], self.track_frames_[:,1] = self.locx, self.locy

            # If not curled
            if len(endpts)==2:
                self.tail_ = endpts[1]
                self.tail_ang = endpt_angles[1]
                self.align_worm()
            else: # If curled, there is only one endpoint.
                self.curl = True
                self.tail_ = endpts[0]
                self.tail_ang = endpt_angles[0]



        # Else, fill in variables as usual.
        else:
            self.old_head_, self.old_tail_ = self.head_, self.tail_
            self.head_ang = endpt_angles[0]
            self.head_ = endpts[0]

            # if curled
            if len(endpts)==1:
                self.curl = True 
                self.tail_ = endpts[0]
                self.tail_ang = endpt_angles[0]
            # otherwise, align animal
            else:
                self.curl = False
                self.tail_ = endpts[1]
                self.tail_ang = endpt_angles[1]

                # Check that body orientation is correct relative to HT angles
                self.align_worm()
                # make sure old and new endpoints are same relative to each other; if not switch
                self.check_switch_HT([self.head_,self.tail_],[self.old_head_,self.old_tail_])

            # Update previous locations
            self.track_frames_[:-1,:] = self.track_frames_[1:,:]
            self.track_frames_[-1,:] = [self.locx, self.locy]

            # Get velocity
            self.old_mvt_ang_ = self.mvt_ang
            travel_vec = self.track_frames_[-1,:] - self.track_frames_[0,:]
            self.speed = np.linalg.norm(travel_vec) / self.track_frames_len_
            if self.speed > 10:
            	self.speed = 0
            self.mvt_ang = worm_bound(np.arctan2(*np.flip(travel_vec))*(180/np.pi))

                    
            # Check if animal is on average moving in a direction opposite its body angle
            self.HT_sum_[:-1] = self.HT_sum_[1:] # as usual, 0th index is the oldest 
            self.HT_sum_[-1] =  projection(self.body_ang, self.mvt_ang, self.speed)
            if (np.sum(self.HT_sum_) < 0) and self.speed>self.speed_th:
                self.HT_sum_ *= -1
                print('Flipping',self.im_num)
                self.flip_worm()

        try:
            self.body_ang = self.body_ang[0]
        except TypeError:
            pass



    def flip_worm(self):
        # Assumes HT endpoints and body orientation are aligned.
        self.head_, self.tail_ = self.tail_, self.head_ 
        self.head_ang, self.tail_ang = self.tail_ang, self.head_ang 
        self.body_ang = worm_bound(self.body_ang-180)


    def align_worm(self):
        # 1. Finds angle of vector pointing from tail to head
        HTang = worm_bound(np.arctan2(*np.flip(self.head_-self.tail_))*(180/np.pi))
        # 2. Finds closest angle between that and body
        diff = ang_between(HTang,self.body_ang)
        # 3. If that angle is bigger than pi/2, flip body.
        if diff > 90:
            self.body_ang = worm_bound(self.body_ang-180)

    def check_switch_HT(self,new_endpts,old_endpts):
        # Flips animal if it should be switched.
        new_endpts, old_endpts = np.array(new_endpts), np.array(old_endpts)
        one = np.mean(np.linalg.norm(new_endpts-old_endpts,axis=0))
        two = np.mean(np.linalg.norm(np.flipud(new_endpts)-old_endpts,axis=0))
        if one <= two:
            pass
        else:
            self.flip_worm()

    def reset(self):
        self.start = True
        self.im_num = -1

        self.locx = 0
        self.locy = 0
        self.body_ang = 0
        self.head_ang = 0
        self.tail_ang = 0
        self.speed = 0
        self.mvt_ang = 0
        self.curl = False

        # Coordinates; only used internally
        self.head_ = np.zeros(2)
        self.tail_ = np.zeros(2)

        self.track_frames_ = np.zeros((self.track_frames_len_,2)) #(x,y), first index is in increasing time order
        self.HT_sum_ = np.zeros(self.HT_sum_len) # If the sum of this vector is negative, flip
        self.old_mvt_ang_ = 0

        self.bg = make_bg(self.cam, self.cam_params, bgtime=self.bg_time)

class FakeWorm():
    '''
    A fake worm class that reads in images and returns descriptors.
    The descriptors saved in the class are:
        locx, locy
        body_ang
        head_, tail_ (endpts)
        head_ang, tail_ang
        curl
        rev
        speed
        mvt_ang

    Also of interest:
        track_frames_ (array of last [track] locations)

    '''
    def __init__(self,folder,cam_id=1,track=6,HT_sum_len=45,rev_th=50,make_bg=True,rotation=0):
        # Hyperparameters #
        self.track_frames_len_ = track
        self.HT_sum_len = HT_sum_len # Number of points to track to see when to flip
        self.rev_th = rev_th # Change in mvt angle that will be marked as a reversal or omega turn
        if cam_id==1:
            self.threshold=8
        else:
            self.threshold=40

        # Init stuff #

        self.rotation = rotation
        if rotation==90:
            self.rotator = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif rotation==180:
            self.rotator = cv2.ROTATE_180
        elif rotation==270:
            self.rotator = cv2.ROTATE_90_CLOCKWISE

        self.folder = folder 
        self.templates = iv.load_templates()

        if make_bg:
            self.bg = self.make_bg()
        else:
            self.bg = self.make_bg(start=0,end=0,res=1)

        self.kerns, self.not_kerns = iv.define_endpt_kernels()
        self.mask = iv.make_mask(self.bg)

        self.reset()
        self.im_num = -1


    def __str__(self):
        return f'Worm {self.im_num}: loc {self.locx,self.locy}, endpts {self.head_,self.tail_},\n \
            Body {self.body_ang}, HT {self.head_ang,self.tail_ang},\n \
            Curl {self.curl}, Reversal {self.rev},\n \
            Speed {np.round(self.speed,2)}, Mvt angle {np.round(self.mvt_ang,2)}'

    def step(self,im_num=None):

        '''
        Load image and get endpoints, endpoint angles, skeleton, location. 
        Updated attributes:
            locy,locx,body_ang,head_ang,tail_ang,curl,rev,speed,mvt_ang
        A 0 is a success; 1 is failure
        '''
        if im_num is None:
            self.im_num += 1
        else:
            self.im_num = im_num

        # Getting worm img and location
        self.img = cv2.imread(f'{self.folder}{self.im_num}.jpg',0)
        if self.rotation!=0:
            self.img = cv2.rotate(self.img, self.rotator)

        self.worm, locs = iv.find_worm(self.img,self.mask,self.bg,threshold=self.threshold)
        if self.worm is None:
            print(f'Empty frame {self.im_num}')
            return 1
        self.locy,self.locx = locs
        self.locy *=-1
        try:
            th,skel = iv.get_wormies(self.worm)
        except cv2.error as e:
            print(f'Gaussian blur error frame {self.im_num}')
            return 1
        endpts = iv.skeleton_endpoints(skel,self.kerns,self.not_kerns)
        # If there are too many endpoints (bright interfering spots), skip this frame
        if len(endpts)>2:
            print(f'Passed frame {self.im_num}')
            return 1
        if len(endpts)==0: # Full circle
            self.curl = True
            # Update previous locations
            self.track_frames_[:-1,:] = self.track_frames_[1:,:]
            self.track_frames_[-1,:] = [self.locx, self.locy]
            # Get velocity
            travel_vec = self.track_frames_[-1,:] - self.track_frames_[0,:]
            self.speed = np.linalg.norm(travel_vec) / self.track_frames_len_
            self.mvt_ang = worm_bound(np.arctan2(*np.flip(travel_vec))*(180/np.pi))
            return 1

        self.body_ang = iv.get_body_angle(skel)
        endpt_angles = iv.get_HT_angles(iv.get_HT_ims(th,endpts),self.templates)
        
        # Fix flip and negative y image issue for endpoints
        endpts = np.fliplr(endpts)
        endpts[:,1] *= -1

        # If it's the first step, initialize some variables.
        # Note: all the flips are for switching x and y variables from image.
        if self.start:
            self.start = False
            self.head_ = endpts[0]
            self.head_ang = endpt_angles[0]

            # If first step, fill in previous locations for speed tracking
            self.track_frames_[:,0], self.track_frames_[:,1] = self.locx, self.locy

            # If not curled
            if len(endpts)==2:
                self.tail_ = endpts[1]
                self.tail_ang = endpt_angles[1]
                self.align_worm()
            else: # If curled, there is only one endpoint.
                self.curl = True
                self.tail_ = endpts[0]
                self.tail_ang = endpt_angles[0]



        # Else, fill in variables as usual.
        else:
            self.old_head_, self.old_tail_ = self.head_, self.tail_
            self.head_ang = endpt_angles[0]
            self.head_ = endpts[0]

            # if curled
            if len(endpts)==1:
                self.curl = True 
                self.tail_ = endpts[0]
                self.tail_ang = endpt_angles[0]
            # otherwise, align animal
            else:
                self.curl = False
                self.tail_ = endpts[1]
                self.tail_ang = endpt_angles[1]

                # Check that body orientation is correct relative to HT angles
                self.align_worm()
                # make sure old and new endpoints are same relative to each other; if not switch
                self.check_switch_HT([self.head_,self.tail_],[self.old_head_,self.old_tail_])

            # Update previous locations
            self.track_frames_[:-1,:] = self.track_frames_[1:,:]
            self.track_frames_[-1,:] = [self.locx, self.locy]

            # Get velocity
            self.old_mvt_ang_ = self.mvt_ang
            travel_vec = self.track_frames_[-1,:] - self.track_frames_[0,:]
            self.speed = np.linalg.norm(travel_vec) / self.track_frames_len_
            if self.speed > 5:
                self.speed = 0
            self.mvt_ang = worm_bound(np.arctan2(*np.flip(travel_vec))*(180/np.pi))

            # Detect reversal: conditions are that movement angle change is at least some 
            diff = ang_between(self.mvt_ang, self.old_mvt_ang_)
            # If movement angle has suddenly flipped, mark a reversal or omega bend 
            # (whatever this is tracking I guess)
            if np.abs(diff) >= self.rev_th:
                self.rev = True
            else:
                self.rev = False
                    
            # Check if animal is on average moving in a direction opposite its body angle
            self.HT_sum_[:-1] = self.HT_sum_[1:] # as usual, 0th index is the oldest 
            self.HT_sum_[-1] =  projection(self.body_ang, self.mvt_ang, self.speed)
            if np.sum(self.HT_sum_) < 0:
                self.HT_sum_ *= -1
                print('Flipping',self.im_num)
                self.flip_worm()


        try:
            self.body_ang = self.body_ang[0]
        except TypeError:
            pass
        return 0


    def flip_worm(self):
        # Assumes HT endpoints and body orientation are aligned.
        self.head_, self.tail_ = self.tail_, self.head_ 
        self.head_ang, self.tail_ang = self.tail_ang, self.head_ang 
        self.body_ang = worm_bound(self.body_ang-180)


    def align_worm(self):
        # 1. Finds angle of vector pointing from tail to head
        HTang = worm_bound(np.arctan2(*np.flip(self.head_-self.tail_))*(180/np.pi))
        # 2. Finds closest angle between that and body
        diff = ang_between(HTang,self.body_ang)
        # 3. If that angle is bigger than pi/2, flip body.
        if diff > 90:
            self.body_ang = worm_bound(self.body_ang-180)

    def check_switch_HT(self,new_endpts,old_endpts):
        # Flips animal if it should be switched.
        new_endpts, old_endpts = np.array(new_endpts), np.array(old_endpts)
        one = np.mean(np.linalg.norm(new_endpts-old_endpts,axis=0))
        two = np.mean(np.linalg.norm(np.flipud(new_endpts)-old_endpts,axis=0))
        if one <= two:
            pass
        else:
            self.flip_worm()

    def reset(self):
        self.start = True
        self.im_num = -1

        self.locx = 0
        self.locy = 0
        self.body_ang = 0
        self.head_ang = 0
        self.tail_ang = 0
        self.speed = 0
        self.mvt_ang = 0
        self.rev = False
        self.curl = False

        # Coordinates; only used internally
        self.head_ = np.zeros(2)
        self.tail_ = np.zeros(2)

        self.track_frames_ = np.zeros((self.track_frames_len_,2)) #(x,y), first index is in increasing time order
        self.HT_sum_ = np.zeros(self.HT_sum_len) # If the sum of this vector is negative, flip
        self.old_mvt_ang_ = 0

    def make_bg(self,start=None,end=None,res=None):
        num_ims = len([name for name in os.listdir(self.folder) if os.path.isfile(f'{self.folder}{name}')])-17
        if start is None:
            start,end,res = 0,num_ims,10
        spacing = num_ims//res

        bg = cv2.imread(f'{self.folder}0.jpg').astype('float64')
        for i,ii in enumerate(np.arange(start,end,spacing)):
            bg -= 1/(i+1)*(cv2.subtract(bg,cv2.imread(f'{self.folder}{ii}.jpg').astype('float64')))
        bg = bg.astype('uint8')[:,:,0]
        if self.rotation!=0:
            bg = cv2.rotate(bg, self.rotator)
        return bg