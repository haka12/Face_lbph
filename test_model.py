#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os 
import cv2
from predict import predict,name

path='/home/baka/face_recog/data/test/'
folders=os.listdir(path)
count=0
total_files=0
def test_model():
    global count,total_files
    for folder in folders:
        label = int(folder)
        test_image_path = path + folder
        dirs = os.listdir(test_image_path)
        total_files=total_files+len(dirs)
        for image in dirs:
            image_path = test_image_path +'/' +image
            #print(image_path)
            test_image = cv2.imread(image_path)
            i,l=predict(test_image)
            if l is not None:#TypeError: 'NoneType' object is not subscriptable_______ corrected
                if label!=l[0]:
                    print("{} was predicted to be {} but is {}".format(image_path,name[l[0]],name[label]))
                    count+=1
            

test_model()
accuracy=(1-(count/total_files))*100
print("The accuracy on the test dataset with {} number of data is {}%".format(total_files,accuracy))
#print(total_files)
#print(count)           