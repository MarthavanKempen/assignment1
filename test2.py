from __future__ import print_function
import numpy as np
import cv2
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import creatingDatasets
import sudoku_img_manipulation
import	os
import pylab as pl
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.set_printoptions(threshold=np.nan)

def from_class_vector_to_number(vector):
    return np.argmax(vector)

def from_vector_predictions_to_numbers(predictions):
    number_of_predictions = np.shape(predictions)[0]
    new_predictions = np.zeros(number_of_predictions)
    for i in range(number_of_predictions):
        prediction = predictions[i]
        new_predictions[i] = from_class_vector_to_number(prediction)
    return new_predictions

def sudoku_predictions(sudoku, predictions, number_indices):
    #Takes a 1d vector of predictions as input (from_vector_predictions_to_numbers())
    sudoku_matrix = np.zeros([9,9])
    for i in range(len(predictions)):
        index = number_indices[i]
        prediction = predictions[i]
        (y, x) = sudoku_img_manipulation.linear_index_to_2d(index)
        sudoku_matrix[y,x] = prediction
    return sudoku_matrix
        

image_name = 'image1004.jpg'
image = cv2.imread(image_name)

model = load_model(os.getcwd() + '/sudoku_model.h5')
(sudoku, number_indices) = sudoku_img_manipulation.return_sudoku(image_name, 0)

numbers = sudoku[number_indices,:]
numbers = numbers.reshape(numbers.shape[0],32, 32,1 )

predictions = model.predict(numbers, batch_size = 128)
predictions = from_vector_predictions_to_numbers(predictions)
sudoku_matrix = sudoku_predictions(sudoku, predictions, number_indices)

print(sudoku_matrix)
pl.figure(1) 
pl.imshow(image)
pl.axis('off')

"""
for number_index in number_indices:
    (y, x) = sudoku_img_manipulation.linear_index_to_2d(number_index)
    image = sudoku[number_index, :]
    model.predict(image, batch_size = 128)

"""