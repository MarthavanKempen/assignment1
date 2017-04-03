# -*- coding: utf-8 -*-

#matplotlib inline
KMP_AFFINITY = disabled
import pylab as pl
import numpy as np
import math
from scipy.spatial import distance as dist
#import Opencv library
try:
    import cv2
    #http://docs.opencv.org/2.4
    print("Successfully imported openCV")
except ImportError:
    print("You must have OpenCV installed")


IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
SUDOKU_SIZE= 9
N_MIN_ACTVE_PIXELS = 50

TRAINING_IMAGE_HEIGHT = 32
TRAINING_IMAGE_WIDTH = 32

#sudoku_original = cv2.imread('images/image9.jpg')
sudoku_original = cv2.imread('image1004.jpg')

#some good parameters:
C = 1
blurSize = 5
dilutionKernel = np.ones((1,1), 'uint8')
"""
To do: 
Loop over various different values for the above parameters. Find the largest 
rectangle, between 50% and 99% of the screen?
First try to just vary the blurSize.
    
"""

def binarizeImage(image, C, blurSize, dilutionKernel):
    image =  cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #adaptiveThreshold doc:
    #http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=adaptive#cv2.adaptiveThreshold
    image = cv2.adaptiveThreshold(image,255,1,1,11,C)
    image = cv2.dilate(image, dilutionKernel, iterations = 3)
    image = cv2.medianBlur(image, blurSize)
    
    
    return image
    
def largestContour(image):
    image, contours0, hierarchy = cv2.findContours(image, 
                                                       cv2.RETR_LIST, 
                                                       cv2.CHAIN_APPROX_SIMPLE)    

    #We try to find the biggest rectangle in the image
    #We assume that the sudoku puzzle has 4 sides and is convex. We also assume
    #that the puzzle is the biggest square in the image.
    size_rectangle_max = 0
    for i in range(len(contours0)):
        #approximate contours to polygons
        approximation = cv2.approxPolyDP(contours0[i], 100, True)   
        #Check if the polygon has 4 sides
        if(not (len (approximation)==4)):
            continue;    
        #is the polygon convex ?
        #if(not cv2.isContourConvex(approximation) ): 
            #continue;
        
        #area of the polygon
        size_rectangle = cv2.contourArea(approximation)
        #store the biggest
        if size_rectangle> size_rectangle_max:
            size_rectangle_max = size_rectangle 
            big_rectangle = approximation
    return big_rectangle

#http://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
#This function orders the corners of the contour in clockwise order
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

#sort the corners to remap the image
def getOuterPoints(rcCorners):
    ar = [];
    ar.append(rcCorners[0,0,:]);
    ar.append(rcCorners[1,0,:]);
    ar.append(rcCorners[2,0,:]);
    ar.append(rcCorners[3,0,:]);
    
    #order the points in clockwise direction :(tl, tr, br, bl)
    ar = order_points(np.array(ar))
    return (ar[0], ar[1], ar[2], ar[3])

#points to remap (square in clockwise order: tl, tr, br, bl)
def createCornerPoints(sudoku_cell_image_height, sudoku_cell_image_width, sudoku_size):
    #Points of the corners of the new image
    total_width = sudoku_size * sudoku_cell_image_width
    total_height = sudoku_size * sudoku_cell_image_height
    points = np.array([
                    np.array([0.0,0.0] ,np.float32),
                    np.array([0.0,0.0] ,np.float32) + np.array([total_width,0.0], np.float32),
                    np.array([0.0,0.0] ,np.float32) + np.array([total_width, total_height], np.float32),
                    np.array([0.0,0.0] ,np.float32) + np.array([0.0,total_height], np.float32),
                    ],np.float32)    
    return points
    
def extract_number(warped_gray, y, x):
    #Extract only the part of the image that corresponds to the cell at position y,x in the warped gray image
    #square -> position x-y
    im_number = warped_gray[y*IMAGE_HEIGHT:(y+1)*IMAGE_HEIGHT][:, x*IMAGE_WIDTH:(x+1)*IMAGE_WIDTH]

    #threshold
    im_number_thresh = cv2.adaptiveThreshold(im_number,255,1,1,15,10)
    im_number_thresh2 = im_number_thresh.copy()
    #delete active pixel in a radius (from center) 
    for i in range(im_number.shape[0]):
        for j in range(im_number.shape[1]):
            dist_center = math.sqrt( (IMAGE_WIDTH/2 - i)**2  + (IMAGE_HEIGHT/2 - j)**2);
            if dist_center > 13:
                im_number_thresh[i,j] = 0;

    n_active_pixels = cv2.countNonZero(im_number_thresh)
    return [im_number, im_number_thresh, n_active_pixels]
    
def find_biggest_bounding_box(im_number_thresh):
    #Find the biggest contour/box in the image (the goal is to capture
    #                                            the number only)
    im_number_thresh, contour,hierarchy = cv2.findContours(im_number_thresh.copy(),
                                         cv2.RETR_CCOMP,
                                         cv2.CHAIN_APPROX_SIMPLE)

    biggest_bound_rect = [];
    bound_rect_max_size = 0;
    for i in range(len(contour)):
         bound_rect = cv2.boundingRect(contour[i])
         size_bound_rect = bound_rect[2]*bound_rect[3]
         if  size_bound_rect  > bound_rect_max_size:
             bound_rect_max_size = size_bound_rect
             biggest_bound_rect = bound_rect
    #bounding box a little more bigger
    x_b, y_b, w, h = biggest_bound_rect;
    x_b= x_b-1;
    y_b= y_b-1;
    w = w+2;
    h = h+2; 
    return [x_b, y_b, w, h]

def insert_biggest_bounding_box(image):
    #Inserts the smaller image into the center of the larger empty image
    newImage = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH])
    h, w = image.shape[:2]
    height_diff = IMAGE_HEIGHT - h
    width_diff = IMAGE_WIDTH - w
    
    vertical_offset = int(height_diff / 2)
    horizontal_offset = int(width_diff / 2)
    
    newImage[vertical_offset:vertical_offset + h, horizontal_offset:horizontal_offset + w] = image
    return newImage
    
def recognize_number(warped_gray, sudoku, y, x):
    #Modifies the sudoku array
    
    #Recognize the number in the rectangle    
    #extract the number (small squares) at position x,y in the warped image
    #and put it in the sudoku matrix. If no number is found, insert an empty image
    #into the sudoku matrix
    [im_number, im_number_thresh, n_active_pixels] = extract_number(warped_gray, y, x)

    if n_active_pixels> N_MIN_ACTVE_PIXELS:
        #find biggest box surrounding the number at position x,y
        [x_b, y_b, w, h] = find_biggest_bounding_box(im_number_thresh)

        im_t = cv2.adaptiveThreshold(im_number,255,1,1,15,9);
        #Retrieve the sub-image within the bounding box
        number = im_t[y_b:y_b+h, x_b:x_b+w]
        #pl.figure(7)
       # pl.imshow(number)

        if number.shape[0]*number.shape[1]>0:            
            #Insert the box bounding the number into the center of a larger empty image            
            number = insert_biggest_bounding_box(number)
         #   number = cv2.resize(number, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
         #   ret,number2 = cv2.threshold(number, 127, 255, 0)
            
            sudoku[y*9+x,:] = number
            #sudoku[x*9+y, :] = number;
            return 1
            

    
    #No number was found. Inserting empty image
    sudoku[y*9+x, ] = np.zeros([IMAGE_WIDTH,IMAGE_HEIGHT]);
    return 0   

def fill_sudoku(warped_gray):
    sudoku  = np.zeros([SUDOKU_SIZE * SUDOKU_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH])
    number_indices = []
    for x in range(SUDOKU_SIZE):
        for y in range(SUDOKU_SIZE):
            foundNumber = recognize_number(warped_gray, sudoku, y, x)
            if foundNumber:
                index = 9*y + x
                number_indices.append(index)
    number_indices = np.sort(number_indices)
    return (sudoku, number_indices)

def resize_sudoku(sudoku):
    resized_sudoku = np.zeros([np.shape(sudoku)[0], TRAINING_IMAGE_HEIGHT, TRAINING_IMAGE_WIDTH])
    for i in range(np.shape(sudoku)[0]):
        number = sudoku[i,:]
        number = cv2.resize(number, (TRAINING_IMAGE_WIDTH, TRAINING_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        ret,number = cv2.threshold(number, 127, 255, 0)
        resized_sudoku[i,:] = number
    return resized_sudoku

def linear_index_to_2d(index):
    y = int(index /SUDOKU_SIZE)
    x = int(index % SUDOKU_SIZE)
    return (y,x)
def return_sudoku(image_name, printing = 0):
    
    sudoku_original = cv2.imread(image_name)
    if printing:
        pl.figure(1)
        pl.imshow(sudoku_original)
        pl.axis('off')
    
    #Binarize the image
    sudoku_binary = binarizeImage(sudoku_original, C, blurSize, dilutionKernel)
    
    if printing:
        pl.figure(1)
        pl.imshow(sudoku_binary, cmap = pl.gray())
        pl.axis('off')
    
    #Show the best candidate 
    big_rectangle = largestContour(sudoku_binary)
    
    #size of the image(height, width)
    h, w = sudoku_original.shape[:2]
    contour_area = cv2.contourArea(big_rectangle)
    area_ratio = contour_area / (h*w)
    if printing:
        sudoku_candidates = sudoku_original.copy()
        for i in range(len(big_rectangle)):
            cv2.line(sudoku_candidates,
                     (big_rectangle[(i%4)][0][0], big_rectangle[(i%4)][0][1]), 
                     (big_rectangle[((i+1)%4)][0][0], big_rectangle[((i+1)%4)][0][1]),
                     (255, 0, 0), 5)

        #show image
        pl.figure(2)
        pl.imshow(sudoku_candidates, cmap=pl.gray()) 
        pl.axis("off")
    
    
    #points to remap (square in clockwise order: tl, tr, br, bl)             
    points1 = createCornerPoints(IMAGE_HEIGHT, IMAGE_WIDTH, SUDOKU_SIZE)                  
    outerPoints = getOuterPoints(big_rectangle)
    points2 = np.array(outerPoints,np.float32)
    
    #Transformation matrix
    pers = cv2.getPerspectiveTransform(points2,  points1 );
    
    #remap the image
    warp = cv2.warpPerspective(sudoku_original, pers, (SUDOKU_SIZE*IMAGE_HEIGHT, SUDOKU_SIZE*IMAGE_WIDTH));
    warped_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    
    if printing:
        #show image
        pl.figure(3)
        pl.imshow(warped_gray, cmap=pl.gray())
        pl.axis("off")
    
    sudoku  = np.zeros([9*9, IMAGE_WIDTH, IMAGE_HEIGHT])
    (sudoku, number_indices) = fill_sudoku(warped_gray)
    sudoku = resize_sudoku(sudoku)
    
    if printing:
        pl.figure(4)
        for index in range(0, SUDOKU_SIZE * SUDOKU_SIZE):
        
            pl.subplot(9,9,index + 1)
            pl.axis('off')
            pl.imshow(sudoku[index,:])
   # x = 2
  #  y = 0
   # index = 9*y + x
  #  pl.figure()
  #  pl.imshow(sudoku[index,:])
    return (sudoku, number_indices)
#return_sudoku()

"""
#sudoku representation
sudoku = np.zeros(shape=(9*9,IMAGE_WIDTH*IMAGE_HEIGHT))

def Recognize_number(warped_gray, x, y):
    
    #Recognize the number in the rectangle    
    #extract the number (small squares)
    [im_number, im_number_thresh, n_active_pixels] = extract_number(warped_gray, x, y)

    if n_active_pixels> N_MIN_ACTVE_PIXELS:
        [x_b, y_b, w, h] = find_biggest_bounding_box(im_number_thresh)

        im_t = cv2.adaptiveThreshold(im_number,255,1,1,15,9);
        number = im_t[y_b:y_b+h, x_b:x_b+w]
        #pl.figure(7)
       # pl.imshow(number)
        print number.shape
        if number.shape[0]*number.shape[1]>0:
            number = cv2.resize(number, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            ret,number2 = cv2.threshold(number, 127, 255, 0)
            number = number2.reshape(1, IMAGE_WIDTH*IMAGE_HEIGHT)
            sudoku[x*9+y, :] = number;
            return 1
            
        else:
            #No number was found, inserting empty image
            sudoku[x*9+y, :] = np.zeros(shape=(1, IMAGE_WIDTH*IMAGE_HEIGHT));
            return 0
            
    else:
        #No number was found. Inserting empty image
        sudoku[x*9+y, :] = np.zeros(shape=(1, IMAGE_WIDTH*IMAGE_HEIGHT));
        return 0

Recognize_number(warp_gray, 0, 2)

index_subplot=0
n_numbers=0
indexes_numbers = []
for i in range(SUDOKU_SIZE):
    for j in range(SUDOKU_SIZE):
        if Recognize_number(warp_gray, i, j)==1:
            if (n_numbers%5)==0:
                index_subplot=index_subplot+1
            indexes_numbers.insert(n_numbers, i*9+j)
            n_numbers=n_numbers+1

pl.figure(7)
pl.imshow(sudoku)
pl.axis('off')
print np.shape(sudoku)


#create subfigures
f,axarr= pl.subplots(index_subplot,5)

width = 0;
for i in range(len(indexes_numbers)):
    ind = indexes_numbers[i]
    if (i%5)==0 and i!=0:
        width=width+1
    axarr[i%5, width].imshow(cv2.resize(sudoku[ind, :].reshape(IMAGE_WIDTH,IMAGE_HEIGHT), (IMAGE_WIDTH*5,IMAGE_HEIGHT*5)), cmap=pl.gray())
    axarr[i%5, width].axis("off")

"""


