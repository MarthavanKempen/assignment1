
import os
import cv2
import numpy as np
from scipy.ndimage import imread

img_h = 32
img_w = 32

folders = ['\sample002', '\sample003', '\sample004', '\sample005', '\sample006',
        '\sample007', '\sample008', '\sample009', '\sample010']
old_dir = os.getcwd()
font_dir = "F:\Browser Downloads\EnglishFnt\English\Fnt"
def load74KData(font_dir):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for (folder, digit) in zip(folders, range(1,10)):
        digit_font_dir = font_dir +  folder
        os.chdir(digit_font_dir)
        
        current_dir = os.getcwd()
        
        image_names = os.listdir(current_dir)
        
        num_images = len(image_names)
        num_training = int(0.9*num_images)
        num_test = num_images - num_training
        
        X_digit_training = np.zeros([num_training, img_h, img_w])
        X_digit_test = np.zeros([num_test, img_h, img_w])
        
        Y_digit_training = np.ones(num_training) * digit
        Y_digit_test = np.ones(num_test) * digit
        
        for i in range(0, num_images):
            image_name = image_names[i]
            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE )
            image = 255-image
           # ret, image = cv2.threshold(image, 127, 255, 0)
            if i < num_training :
                X_digit_training[i,:] = image
            else:
                X_digit_test[i-num_training,:] = image

        X_train.append(X_digit_training)
        X_test.append(X_digit_test)
        Y_train.append(Y_digit_training)
        Y_test.append(Y_digit_test)
    
    X_train = np.concatenate(X_train, axis = 0)
    X_test = np.concatenate(X_test, axis = 0)
    Y_train = np.concatenate(Y_train, axis = 0)
    Y_test = np.concatenate(Y_test, axis = 0)
    
    os.chdir(old_dir)
    return ((X_train, Y_train), (X_test, Y_test))

#print(np.shape(X_train))
#print(np.shape(Y_train))
#print(np.shape(X_test))
#print(np.shape(Y_test))