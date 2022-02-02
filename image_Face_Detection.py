# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 14:12:03 2022

@author: Palla
"""

#importing the required libraries
import cv2
import face_recognition

image_to_detect=cv2.imread('images/samples/harshit.jpg')

#cv2.imshow("trial",image_to_detect )

#print the number of faces detected
all_face_locations=face_recognition.face_locations(image_to_detect,model='hog')

#print the number of faces detected
print('There are {} no of faces in this image'.format(len(all_face_locations)))
#place holder -> {}

#looping through the face locations
for index,current_face_location in enumerate(all_face_locations):
    #splitting the tuple to get the four positions values of current face
    top_pos,right_pos,bottom_pos,left_pos=current_face_location
    #printing the locations of currnet face
    print('Found face {} at top:{},right:{},bottom: {},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
    #slicing the current face from main image
    current_face_image=image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    #showing the current face with dynamic title
    cv2.imshow("Face no "+str(index+1),current_face_image)



