import numpy as np
import cv2
import matplotlib.pyplot as plt
import mahotas as mh

from skimage.morphology import skeletonize
from skimage import data, img_as_bool
import matplotlib.pyplot as plt
from skimage.util import invert
import numpy as np
import cv2
import matplotlib.pyplot as plt
from fil_finder import FilFinder2D
import astropy.units as u
import networkx as nx
from scipy import ndimage

def endPoints(skel):
    # Input: 
    # Skeletonized image
    #
    # Output: 
    # Endpoints of skeleton via morphological transformation
    
    endpoint1=np.array([[0, 0, 0],
                        [0, 1, 0],
                        [2, 1, 2]])
    
    endpoint2=np.array([[0, 0, 0],
                        [0, 1, 2],
                        [0, 2, 1]])
    
    endpoint3=np.array([[0, 0, 2],
                        [0, 1, 1],
                        [0, 0, 2]])
    
    endpoint4=np.array([[0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 0]])
    
    endpoint5=np.array([[2, 1, 2],
                        [0, 1, 0],
                        [0, 0, 0]])
    
    endpoint6=np.array([[1, 2, 0],
                        [2, 1, 0],
                        [0, 0, 0]])
    
    endpoint7=np.array([[2, 0, 0],
                        [1, 1, 0],
                        [2, 0, 0]])
    
    endpoint8=np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 2, 0]])
    
    ep1=mh.morph.hitmiss(skel,endpoint1)
    ep2=mh.morph.hitmiss(skel,endpoint2)
    ep3=mh.morph.hitmiss(skel,endpoint3)
    ep4=mh.morph.hitmiss(skel,endpoint4)
    ep5=mh.morph.hitmiss(skel,endpoint5)
    ep6=mh.morph.hitmiss(skel,endpoint6)
    ep7=mh.morph.hitmiss(skel,endpoint7)
    ep8=mh.morph.hitmiss(skel,endpoint8)
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    
    return ep


def eucToMat(x,y,n):
    # Input:
    # x: image index of non zero x values
    # y: image index of non zero y values
    # n: size of square output image
    #
    # Output:
    # Single worm on a numpy array
    
    mask = np.zeros((n, n))
    
    col = x
    row = n - 1 - y 

    for matLoc in zip(row,col):
        mask[matLoc[0],matLoc[1]] = 1
    
    return img_as_bool(mask)

def defineHood(x, bound):
    # Input:
    #     
    #
    # Output:
    #
    #
    
    l = []
    if x - 1 < 0:
        l = list(range(x,x+2))
    elif x + 1 > bound-1:
        l = list(range(x-1,x+1))
    else:
        l = list(range(x-1, x+2))
    
    return l


def Skel2Graph(G,x,y,skeleton):
    # Input:
    #
    #
    # Output: Inplace
    #
    #
    
    rows = defineHood(x, skeleton.shape[0])
    cols = defineHood(y, skeleton.shape[1])
    for r in rows:
        for c in cols:
            if skeleton[r, c] == True and (r != x or c != y):
                G.add_edge((x,y), (r,c))
                
def findLongestPath(G, endpoints):
     # Input:
    #
    #
    # Output:
    #
    #
    
    numEnds = len(endpoints[0])
    if numEnds < 2:
        return None
    
    pathLengths = []

    for i in range(numEnds):
        for j in range(i,numEnds):
            if i != j:
                src = (endpoints[0][i],endpoints[1][i])
                targ = (endpoints[0][j],endpoints[1][j])
                pathLengths.append(nx.shortest_path(G, source=src, target=targ))
                
    longPath = []
    maxlen = 0

    for path in pathLengths:
        n = len(path)
        if n > maxlen:
            maxlen = n
            longPath = path
                        
    return longPath

            
def calcHeadAngles(path):
    # Input:
    #
    #
    # Output:
    #
    #
    
    vec01 = np.asarray(path[0]) - np.asarray(path[1])
    vec02 = np.asarray(path[0]) - np.asarray(path[2])
    vec03 = np.asarray(path[0]) - np.asarray(path[3])

    avgVec1 = .2*vec01+.2*vec02+.6*vec03
    
    vec12 = np.asarray(path[-1]) - np.asarray(path[-2])
    vec13 = np.asarray(path[-1]) - np.asarray(path[-3])
    vec14 = np.asarray(path[-1]) - np.asarray(path[-4])
    
    avgVec2 = .2*vec12+.2*vec13+.6*vec14

    # all degrees are relative to the positive x-axis
    
    # arctan2 automatically does x and y swap
    # need to negate the x position to get a vertical y flip
    endAngle1 = np.round(np.degrees(np.arctan2(-avgVec1[0], avgVec1[1])))
    endAngle2 = np.round(np.degrees(np.arctan2(-avgVec2[0], avgVec2[1])))
    
    return endAngle1, endAngle2


bg = cv2.imread("background.jpg",0)
img = cv2.imread("./LawnLeaving/img2.jpg",0)            

def getHeadBodyAngles(bg, img):
    # Input:
    #
    #
    # Output:
    #
    #
    
    
    # subtracting background from image
    imgSub = cv2.subtract(img, bg)

    # applying threshold and finding contours
    _, thresh = cv2.threshold(imgSub, 8, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # identifying worms by contour area > 50
    wormContours = []
    contAreas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        contAreas.append(area)
        if area > 60:
            wormContours.append(contour)

    angles = []
    for cnt in wormContours:
        
        # find center of worm
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        # fit line to worm
        # can calculate slope of fitted line using vy/vx    
        vx,vy,x,y = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)

        # calculate body angle and centroid location
        pos = (cx, cy)
        bodyAngle = np.round(np.degrees(np.arctan2(vy,vx)))[0] + 90
        
        # fill contour and find location of nonzero points relative to original image
        mask = np.zeros(imgSub.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [cnt], -1, 255, -1)        
        loc = np.nonzero(mask)
        
        # extract points and make a mask for skeletonization
        x = loc[1]
        y = loc[0]
        
        minx = min(x)
        miny = min(y)

        x -= minx
        y -= miny

        n = max(max(x), max(y)) + 2
        wormSeg = eucToMat(x, y, n)
        
        # perform skeletonization
        skeleton = np.flip(skeletonize(wormSeg),0)
        skelPoints = skeleton.nonzero()
        
        # get longest path of skeleton
        G = nx.Graph()

        for x,y in zip(skelPoints[0], skelPoints[1]):
            Skel2Graph(G,x,y,skeleton)

        eps = endPoints(skeleton*1).nonzero()

        longPath = findLongestPath(G, eps)  
        
        if longPath:
            # calculate angles based on longest path
            headAng1, headAng2 = calcHeadAngles(longPath)

            e1 = longPath[0]
            e2 = longPath[-1]
            endCoords = np.flip(np.array([e1, e2]),1) + np.array([minx, miny])
            nodes = np.flip(np.array(longPath),1) + np.array([minx, miny])
            
            angles.append([pos, bodyAngle, headAng1, headAng2, nodes])
        else:
            angles.append([pos, bodyAngle, 'bad'])
    
    return angles, wormContours

angs, wormContours = getHeadBodyAngles(bg, img)