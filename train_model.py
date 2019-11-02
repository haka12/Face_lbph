import cv2
import numpy as np
import os

path='/home/baka/face_recog/data/train/'

def face_detection(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cas = cv2.CascadeClassifier('/home/baka/face_recog/cascade/haarcascade_frontalface_default.xml')
    face=cas.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    try:#indexError due to null return from cascade classifier___________correction
        (x,y,w,h)=face[0]
    except IndexError as e:
        return None, None
    return gray[y:y+w,x:x+h],face

def resize(image):
    resized_image = cv2.resize(image, (250,125)) 
    return resized_image

def preprocess_data(path):
    folders=os.listdir(path)
    faces=[]
    labels=[]

    for folder in folders:
        label = int(folder)
        train_image_path = path + folder
        print("travesing through"+train_image_path)
        for image in os.listdir(train_image_path):
            image_path = train_image_path +'/' +image
            training_image = cv2.imread(image_path)
            roi,face = face_detection(training_image)
            if roi is not None:#error: (-215) ssize.width > 0 && ssize.height > 0 in function resize_____________correction
                roi=resize(roi)
                faces.append(roi)
                labels.append(label)
    return faces,labels
           

def model_train():
    faces,labels = preprocess_data(path)
    model=cv2.face.createLBPHFaceRecognizer()
    model.train(faces,np.array(labels))
    model.save('trained.xml')

#uncomment to train model
#model_train()
