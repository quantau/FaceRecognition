# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:29:50 2022

@author: Palla
"""

#importing the required libraries
import cv2
import face_recognition

#capture video stream
webcam_video_stream=cv2.VideoCapture(0)
#if i want to use it over a video then simply change 0 with location of video
#wrt to code

#initialize the array variables
all_face_locations=[]

while True:
    #get current frame
    ret,current_frame=webcam_video_stream.read()
    #resize the frame to a quarter of size so that the
    #computer can process it faster
    current_frame_small=cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #store all face locations
    all_face_locations=face_recognition.face_locations(current_frame_small,model='hog')
    
    for index,current_face_location in enumerate(all_face_locations):
        top_pos,right_pos,bottom_pos,left_pos=current_face_location
        top_pos=top_pos*4
        right_pos=right_pos*4
        left_pos=left_pos*4
        bottom_pos=bottom_pos*4
        print('Found face {} at top:{},right:{},bottom: {},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        #current_face_image=image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
        cv2.imshow("Webcam Feed",current_frame)
    
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break

#release the webcame resource

webcam_video_stream.release()
cv2.destroyAllWindows()