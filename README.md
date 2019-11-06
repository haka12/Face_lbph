# face_recog
Face Recognition using OpenCV

# requirements 
numpy
opencv-utils
opencv-python

# Prepare the dataset
You can prepare the dataset yourself by downloading the images from google.
While preparing the dataset you need to create two datasets one for training and other for 
testing. The split can be any way you like, but its generally a good idea to train on larger 
split than to test them
Make different folders for distinct people and label them numerically.


# Training the Model
Change the path variable that points towards the dataset.
Uncomment model_train function call and run train_model.py to train the model on the dataset.

# Prediction
The trained model is saved in a .xml file in the working directory the prediction is made by loading the file.
uncomment the lines 34-40 to test an individual image to see if the prediction is working or not
You need to edit the dictionary object name in the file predict.py to suit your dataset.

# Testing 
Change the path variable to the test dataset and test the accuracy of your model.
