import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

# to get ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# to get preset data
X, y = fetch_openml('mnist_784', version = 1, return_X_y = True)
print(pd.Series(y).value_counts())

# making classes
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nclasses = len(classes)

# test & train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

# y_pred
clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

# measuring accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# initialize camera
cap = cv2.VideoCapture(0)
while (True):
    try:
        ret, frame = cap.read()

        # make it gray so that the computer doesn't get confused with colors
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # making a box in center - region of interest (roi)
        height, width = gray.shape()
        upperLeft = (int(width/2 - 56), int(height/2 - 56))
        bottomRight = (int(width/2 + 56), int(height/2 + 56))
        cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 0), 2) #to select the frame, then dimensions, then color, then thickness
        roi = gray[upperLeft[1]: bottomRight[1], upperLeft[0]: bottomRight[0]]

        # making pil image 
        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert('L') #to make sure it isn't in any other color
        image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter) #converting to scalar quantity
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted - min_pixel, 0, 255) #using the clip to limit the values
        max_pixel = np.max(image_bw_resized_inverted) #getting the maximum of all the given numbers
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel #creating each frame into an array

        # making the prediction
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1784)
        test_prediction = clf.predict(test_sample) #logistic regression
        print('Predicted number is ', test_prediction)

        # stopping camera
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    except Exception as E:
        pass

# closing the camera 
cap.release()
cv2.destroyAllWindows()