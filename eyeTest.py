import cv2
import sys
import numpy as np
import dlib
import imutils

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector() 


# These are offsets for the each eye
rT, rB, rL, rR = 1,-2,3,-2
lT, lB, lL,lR = 2,-2,3,-3
# the lower xdark, the darker the black it will have to be to get detected, higher value = detects more 
rdarkStart, ldarkStart = 15, 9.5
rdark, ldark = np.array([]), np.array([]) # adjusts lighting for each face 

SAMPLESIZE, minPoints, maxPoints = 30, 10, 15 # all used in samples() for determining point frequency

rpoints = np.array([])
lpoints = np.array([])

video_capture = cv2.VideoCapture(1)


def eye(area, darkness, name='Undefined', drawBorder=True):
    if np.size(area) == 0:
        return 0
    gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, darkness, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    # finding extrema in the eye
    c = max(contours, key=cv2.contourArea)
    c = np.squeeze(c) # reshapes c from (x,1,2) to (x,2)
    extrema = np.array([tuple(getMinMin(c,0,1))]) # bottom right 
    extrema = np.vstack([extrema,tuple(getMinMax(c,0,1))]) # bottom left
    extrema = np.vstack([extrema,tuple(getMaxMax(c,1,0))]) # top right
    extrema = np.vstack([extrema,tuple(getMinMax(c,1,0))]) # top left
    
    # draw the center of the rectangle, should be the pupil
    center = tuple(extrema.mean(0).astype(int))
    cv2.circle(area, center, center[1], (255, 255, 0), 1)
    
    if drawBorder:
        cv2.drawContours(area, [c], -1, (0, 255, 255), 1)

    return len(c) # need to correct for sampling to work correctly
    
def debugArea(area, name = 'Debug Frame'):
    cv2.imshow(name, area)

# d1, d2 are tuples of the form (dimension, name)
def dimCheck(d1, d2):
    if d1[0] == d2[0]:
        raise ValueError(f'Both dimensions can not be the same: {d1[1]} = {d1[0]}, {d2[1]} = {d2[0]}')

# returns the value with Minimum (minDim) and maximum of (maxDim). 
# Example: getMinMax(np.array([[1,0],[3,0],[12,1]]), minDim = 1, maxDim = 0) -> returns [3,0] 
def getMinMax(arr, minDim, maxDim):
    dimCheck((minDim, 'minDim'), (maxDim, 'maxDim'))
    # all indicies that have same value along given dimension 
    minD1 = np.where(arr[:,minDim] == arr[np.argmin(arr[:,minDim])][minDim])[0]
    
    # now get max of other dim (max values for all elements with the smallest element)
    return arr[minD1[np.argmax(arr[minD1][:,maxDim])]]
 
# returns the value with Minimum (dim1) and min of (dim2)
def getMinMin(arr, dim1, dim2):
    dimCheck((dim1, 'dim1'), (dim2, 'dim2'))
    minD1 = np.where(arr[:,dim1] == arr[np.argmin(arr[:,dim1])][dim1])[0]
    return arr[minD1[np.argmin(arr[minD1][:,dim2])]]

# see getMinMin for details, this function is just its inverse
def getMaxMax(arr, dim1, dim2):
    dimCheck((dim1, 'dim1'), (dim2, 'dim2'))
    minD1 = np.where(arr[:,dim1] == arr[np.argmax(arr[:,dim1])][dim1])[0]
    return arr[minD1[np.argmax(arr[minD1][:,dim2])]]

# def eye(area, darkness, name='Undefined'):
#     if np.size(area) == 0:
#         return 0
#     gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
#     ret,thresh = cv2.threshold(gray, darkness, 255, cv2.THRESH_BINARY_INV)
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # contours can track # of points
#     cv2.drawContours(area, contours, -1,(0,0,255),3)
#     #pupil = cv2.drawContours(area, contours, -1,(0,0,255),3)
#     #cv2.imshow(name,pupil)
#     return np.shape(contours)[0]

# adjust rdark and ldark until you desired number of points reached, amount is amount you increment xdark by
def adjustLighting(rightEye, amount, i):
    global rdark
    global ldark
    
    if rightEye:
        rdark[i] += amount 
        print('Right', rdark[i])
    else:
        ldark[i] += amount
        print('Left', ldark[i])
    return

# given number of points (lpoints or rpoints) and sampleSize (int), determine if we should adjust lighting
# rightEye is boolean; 0 for left eye, 1 for right eye
# i is the index for the face we are currently looking at, used in adjustLighting
def samples(points, sampleSize, rightEye, i):
    if len(points) < sampleSize:
        return points
    mean = np.mean(points)  

    # mean is average number of points in a given sample size, adjust lighting if too few points or too many
    if mean < minPoints:
        #print('Right','mean:',mean,'Min:',minPoints) if rightEye else print('Left','mean:',mean,'Min:',minPoints)
        adjustLighting(rightEye, 1, i) if not mean < (minPoints//2) else adjustLighting(rightEye, 3, i) #if statement to speed up
    elif mean >= maxPoints:
        #print('Right','mean:',mean,'Max:',maxPoints) if rightEye else print('Left','mean:',mean,'Max:',maxPoints)
        adjustLighting(rightEye, -1, i) if not mean < (minPoints//2) else adjustLighting(rightEye, -3, i)
    return np.array([])
        
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    
    # find the colors within the specified boundaries and apply the mask
    for i in range(len(faces)):
        landmarks = predictor(gray, faces[i])
        
        if len(faces) > len(rdark):
            rdark = np.append(rdark, [rdarkStart])
            ldark = np.append(ldark, [ldarkStart])
            
        # Right Eye 
        rpoints = samples(rpoints, SAMPLESIZE, 1, i)
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))
        
        areaR = frame[center_top[1]+rT:center_bottom[1]+rB,left_point[0]+rL:right_point[0]+rR]
        
        rpoints = np.append(rpoints, eye(areaR, rdark[i], "Right Eye"))
        
        ## Output the right eye seperately
        #cent = midpoint2(int(np.mean(np.array(rightPupil[0]))),int(np.mean(np.array(rightPupil[1]))))
        #circ = cv2.circle(rightPupil,cent,3,(0,0,255),3)
        #cv2.imshow('R',rightPupil)
        
        # mask = cv2.inRange(areaR, lower_white, upper_white)
        # outputR = cv2.bitwise_and(areaR, areaR, mask = mask)
        
        # Left Eye
        lpoints = samples(lpoints, SAMPLESIZE, 0, i)
        left = (landmarks.part(42).x, landmarks.part(42).y)
        right = (landmarks.part(45).x, landmarks.part(45).y)
        top = midpoint(landmarks.part(43), landmarks.part(44))
        bot = midpoint(landmarks.part(47), landmarks.part(46))
        
        areaL = frame[top[1]+lT:bot[1]+lB,left[0]+lL:right[0]+lR]
        
        lpoints = np.append(lpoints, eye(areaL, ldark[i], "Left Eye"))
        
        ## Output the right eye seperately
        # mask = cv2.inRange(areaL, lower_white, upper_white)
        # outputL = cv2.bitwise_and(areaL, areaL, mask = mask)
        # cv2.imshow("L", areaL)
        # cv2.imshow("R",outputR)
        
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()



