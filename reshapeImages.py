# -*- coding: utf-8 -*-

import os
import cv2

folders = ['\sample002', '\sample003', '\sample004', '\sample005', '\sample006',
        '\sample007', '\sample008', '\sample009', '\sample010']

#fnt_dir = "F:\Browser Downloads\EnglishFnt\English\Fnt"

DESIRED_IMAGE_WIDTH = 32
DESIRED_IMAGE_HEIGHT = 32

i = 0 
for folder in folders:
    char_fnt_dir = fnt_dir +  folder
    os.chdir(char_fnt_dir)
    currentDir = os.getcwd()
    imageNames = os.listdir(currentDir)

    for imageName in imageNames:
        image = cv2.imread(imageName)
        image = cv2.resize(image, (DESIRED_IMAGE_WIDTH, DESIRED_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        image =  cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, image = cv2.threshold(image, 127, 255, 0)
        cv2.imwrite(imageName, image)
        
        print(float(i)/9000.0)
        i = i+1
        
