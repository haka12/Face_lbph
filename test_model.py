#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os 
import cv2
from predict import predict

path='/home/baka/face_recog/data/test/'
folders=os.listdir(path)
for folder in folders:
    label = int(folder)
    test_image_path = path + folder
    
    for image in os.listdir(test_image_path):
        image_path = test_image_path +'/' +image
        #print(image_path)
        test_image = cv2.imread(image_path)
        i,l=predict(test_image)
        print(type(l[0]))
        #if l!=label:
            