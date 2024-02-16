#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2 
import cvzone
import time
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot


# In[16]:


detector = FaceMeshDetector(maxFaces=1)
cam = cv2.VideoCapture(0) 
idList = [22, 23, 24, 26, 110,157, 158, 159, 160, 161, 130, 243]
#these are the points that indicate out Points Of Concern arounfd left eye

plotY = LivePlot(600,500,[20,60])
escape_pressed = False  # Flag to track if escape key was pressed

blinkCount = 0 
#to count the number of blinks
count = 0

while not escape_pressed: 
    _, img = cam.read() 
    key = cv2.waitKey(16)
    img = cv2.resize(img , (600, 500))
    img , faces = detector.findFaceMesh(img , draw=False)
    #faces is a list of face meshes
    
    if faces:
        face  = faces[0]
        #points between whose distance is calculated and ratio is found

        leftU = face[159]
        leftD = face[23]
        leftR = face[243]
        leftL = face[130]
        
        #just to point out the points around left eye
        for id in idList:
            cv2.circle(img , face[id] , 1,(255,255,255),cv2.FILLED)
        lenVer, _ = detector.findDistance(leftU , leftD)
        lenHor, _ = detector.findDistance(leftL, leftR)
        
        #ratio is calcluated and 
        ratio = int((lenVer/lenHor)*100)
        
        #cv2.line(img, leftU , leftD, (0, 0, 200) , 3)
        #cv2.line(img, leftL, leftR, (0, 0, 200), 3)
        #print(ratio)
        
        imgPlotg = plotY.update(ratio)
        imgPlotg = cv2.resize(imgPlotg , (600, 500))
        
        #cv2.imshow("Graph",imgPlotg)
        
        #calculate the no. of blinks
        if ratio<35 and count==0:
            blinkCount += 1
            count = 1
        if count != 0:
            count +=1
            if count>10:
                count = 0
        
        #print(blinkCount)
    
        img = cvzone.stackImages([img , imgPlotg], 2,1)
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (50, 50) 
        fontScale = 1
        color = (255, 0, 0) 
        thickness = 2
        
        #Using cv2.putText() method 
        img = cv2.putText(img, f'LeftEye Blinks: {blinkCount}', org, font,fontScale, color, thickness, cv2.LINE_AA) 
        
    cv2.imshow('Camera Feed', img) 
    if key == ord('q') or key == 27:  # Check if 'q' or Esc key is pressed
        escape_pressed = True  # Set the flag to True to exit the loop

cam.release() 
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




