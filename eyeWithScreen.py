import cv2
import numpy as np
import dlib

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

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


# adjust rdark and ldark until you desired number of points reached, amount is amount you increment xdark by
def adjustLighting(rightEye, amount, i):
    global rdark
    global ldark
    
    if rightEye:
        rdark[i] += amount 
    else:
        ldark[i] += amount
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
        adjustLighting(rightEye, 1, i) if not mean < (minPoints//2) else adjustLighting(rightEye, 3, i) #if statement to speed up
    elif mean >= maxPoints:
        adjustLighting(rightEye, -1, i) if not mean < (minPoints//2) else adjustLighting(rightEye, -3, i)
    return np.array([])

    
# gets and draws pupil, returns the number of all points in contour, which is used in samples
def eye(area, darkness, rightEye, drawBorder=False):
    if np.size(area) == 0:
        return 0
    gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, darkness, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.medianBlur(thresh, 3) 
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
    
    if rightEye:
        global rightCenter
        rightCenter = center
    else:
        global leftCenter
        leftCenter = center
    
    if drawBorder:
        cv2.drawContours(area, [c], -1, (0, 255, 255), 1)

    return len(c) # need to correct for sampling to work correctly

# this function gets a sample 
def samplePortion(exitKey):
    global leftCenter, rightCenter
    print('Capturing', '\n')
    meanArrayL, meanArrayR = np.array([]), np.array([])
    
    loopVideo(True, exitKey)
    if not rightCenter == (0,0):
         meanArrayR = np.append(meanArrayR,rightCenter)
    if not leftCenter == (0,0):
        meanArrayL = np.append(meanArrayL,leftCenter)
        
    return (meanArrayL.mean(0).astype(int), meanArrayR.mean(0).astype(int))

# draws 1 point 
def drawPoint(point, color = (0,0,255)):
    global currentFrame
    cv2.circle(currentFrame, point, 1, color, 1)

# might not need this. Can just look eye area and compare the center of the eye to get an idea of where im looking
def setAreas():
    print("Look at top left of screen, f to start, g to end")
    loopVideo(False,'f')
    tL = samplePortion('g')
    print('top left:',tL)

    print("Look at top right of screen, f to start, g to end")
    loopVideo(False,'f')
    tR = samplePortion('g')
    print('top right:',tR)
        
    print("Look at bottom left of screen, f to start, g to end")
    loopVideo(False,'f')
    bL = samplePortion('g')
    print('bottom left:',bL)
    
    print("Look at bottom right of screen, f to start, g to end")
    loopVideo(False,'f')
    bR = samplePortion('g')
    print('bottom right:',bR)
    
    print("Look at center of screen, f to start, g to end")
    loopVideo(False,'f')
    c = samplePortion('g')
    print('center:', c)
    
    return tL, tR, bL, bR, c

def loopVideo(eyeBorder, exitKey, func = None):
    global detector, predictor, ldark, rdark, rdarkStart, ldarkStart, rpoints, lpoints, video_capture
    global currentFrame
    while (True):
        _, frame = video_capture.read()
        
        currentFrame = frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for i in range(len(faces)):
            landmarks = predictor(gray, faces[i])
            if len(faces) > len(rdark):
                    rdark = np.append(rdark, [rdarkStart])
                    ldark = np.append(ldark, [ldarkStart])
            rpoints = samples(rpoints, SAMPLESIZE, 1, i)
            lpoints = samples(lpoints, SAMPLESIZE, 0, i)
            areaL, areaR = computeEyeAreas(frame, landmarks)
            rpoints = np.append(rpoints, eye(areaR, rdark[i], True))
            lpoints = np.append(lpoints, eye(areaL, ldark[i], False))
            
        if cv2.waitKey(1) & 0xFF == ord(exitKey):
            break
        cv2.imshow("Frame", frame)

def loopVideoLight(exitKey):
    global video_capture
    while (True):
        _, frame = video_capture.read()
        if cv2.waitKey(1) & 0xFF == ord(exitKey):
            break
        cv2.imshow("Frame", frame)
    
# trying to get an idea of where im looking based on the over all eye area and where my pupil is in respect to that
# WIP
def lookingLocation(areaL, areaR, leftCenter, rightCenter):
    # only need to look at first 2 dimensions of areaL and areaR, since last dimension is color
    if np.size(areaL) == 0 or np.size(areaR) == 0:
        return 0
    cv2.circle(areaL,leftCenter,1,(0, 255, 0), 0)
    scale_percent = 220 # percent of original size
    width = int(areaL.shape[1] * scale_percent / 100)
    height = int(areaL.shape[0] * scale_percent / 100)
    dim = (1850, 1000)
    # resize image
    resized = cv2.resize(cv2.flip(areaL,1), dim, interpolation = cv2.INTER_AREA) 
    cv2.imshow('L',resized)
    
# given facial landmarks, locates the left and right eye in the frame
def computeEyeAreas(frame, landmarks):
    global offSetsL, offSetsR
    # Right Eye 
    right_left = (landmarks.part(36).x, landmarks.part(36).y)
    right_right = (landmarks.part(39).x, landmarks.part(39).y)
    right_top = midpoint(landmarks.part(37), landmarks.part(38))
    right_bot = midpoint(landmarks.part(41), landmarks.part(40))
    
    areaR = frame[right_top[1]+offSetsR[0]:right_bot[1]+offSetsR[1],right_left[0]+offSetsR[2]:right_right[0]+offSetsR[3]]
    
    # Left Eye
    left_left = (landmarks.part(42).x, landmarks.part(42).y)
    left_right = (landmarks.part(45).x, landmarks.part(45).y)
    left_top = midpoint(landmarks.part(43), landmarks.part(44))
    left_bot = midpoint(landmarks.part(47), landmarks.part(46))
    
    areaL = frame[left_top[1]+offSetsL[0]:left_bot[1]+offSetsL[1],left_left[0]+offSetsL[2]:left_right[0]+offSetsL[3]]
    
    return areaL, areaR


# AREAS ARE A WIP
if __name__ == "__main__":
    # easy to turn into class, take all global variables and make them class variables. That's it
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    
    # These are offsets for the each eye
    offSetsL = [2,-2,3,-3] # lT, lB, lL, lR (left right side offset, left bottom offset, ...)
    offSetsR = [1,-2,3,-2] # rT, rB, rL, rR
   
    # the lower xdark, the darker the black it will have to be to get detected, higher value = detects more 
    rdarkStart, ldarkStart = 20, 15
    rdark, ldark = np.array([]), np.array([]) # adjusts lighting for each face 
    
    leftCenter, rightCenter = (0,0), (0,0)
    
    SAMPLESIZE, minPoints, maxPoints = 30, 10, 15 # all used in samples() for determining point frequency
    
    lpoints, rpoints= np.array([]), np.array([])
    
    video_capture = cv2.VideoCapture(0)
    currentFrame = 0
    #topLeftArea,topRightArea,botLeftArea,botRightArea,centerArea = setAreas()
    
    print('\n', 'Main program started, press q to quit')
    loopVideo(False, 'q')
    
    cv2.destroyAllWindows()
   
