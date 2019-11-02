#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from train_model import face_detection,resize

name={
      1:"Tom Cruise",
      2:"Jennifer Connely",
      3:"Leo Messi",
      4:"Elon Musk",
      5:"Jerry Seinfeld"
      }


def predict(img):
    roi, face = face_detection(img)
    
    if roi is not None:#error: (-215) ssize.width > 0 && ssize.height > 0 in function resize________________correction
        roi=resize(roi)
    else:
        return None,None
    #cv2.imshow('roi',roi)
    model=cv2.face.createLBPHFaceRecognizer()
    model.load('trained.xml')
    label = model.predict(roi)
    #print (label)
    text=(name[label[0]])
    (x,y,w,h)=face[0]
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),1)
    return img,label

# =============================================================================
# test1=cv2.imread('1.jpg')
# img,label=predict(test1)
# #print(label[0])
# cv2.imshow('image1',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================
