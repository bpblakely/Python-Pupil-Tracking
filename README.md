# Python-Pupil-Tracking
Python program to track your pupils in real time

The way this program works is by:

1. Detecting the face
2. Detecting the eyes from the face
3. Contrasting the dark and white areas of the eye (pupil color vs white part of eye)
4. Draws the contours as red dots

**Some Notes**

This program works best if you have darker colored eyes. The lighting around you also impacts the correctness of the tracking, but the program does automatically adjust the lighting detection depending on the average of points found in the pupil. So if in the last X samples there is less than Y samples, it will automatically reduce the darkness threshold to try and pick up the pupil. Likewise, if there are too many points it will start to increase the threshold making lighter colors appear less (such as white). If the threshold is too high then the white areas around the pupil will start to get detected, which is the opposite purpose of this program. Also, this is my first time working with live video and I'm learning everything from scratch.

**Features List / To Do**

- [x] Track the pupil quickly and accurately in live video
- [ ] Map the relative location of your pupil to where you are looking on screen
- [ ] Implement a GUI to show the output and adjust parameters

**Dependencies**

- CV2
- Numpy
- Dlib

**Dlib Required File**

For Dlib face detection to work you need to download the file "shape_predictor_68_face_landmarks.dat" and use that for the predictor. There are many different predictor files you can use, however this is the only one I have tested. 

You can find the download on this link https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
