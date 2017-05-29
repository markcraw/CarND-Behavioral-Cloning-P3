import os
import os.path
import csv
import argparse
import json
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

samples = []
# Load training data it includes images and steering angles for the vehicle driving on the road
with open('./data/driving_orig_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    print('/opt/ros/kinetic/lib/python2.7/dist-packages, not found')

import cv2
import numpy as np
import sklearn

# The image size for the training data is below
ch, row, col = 3, 160, 320   # camera format

# This function builds the CNN model using Keras
def get_model_cai(time_len=1):
  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
  model.add(Cropping2D(cropping=((75,25),(0,0))))
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  #model.add(ELU())
  #model.add(Convolution2D(128, 5, 5, subsample=(2, 2), border_mode="same"))    
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  return model

# The generator function selects batch sizes of inputs to feed to the model during training
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = './data/IMG/'+batch_sample[0].split('/')[-1]
                name_left = './data/IMG/'+batch_sample[1].split('/')[-1]
                name_right = './data/IMG/'+batch_sample[2].split('/')[-1]
                
                # Get the center images from the vehicle and flip it horizontally
                center_image = cv2.imread(name_center)
                center_image_flip = cv2.flip(center_image,1)
                center_angle = float(batch_sample[3])
                center_angle_flip = -center_angle                
                
                # Get the left images from the vehicle and flip it horizontally
                left_image = cv2.imread(name_left)
                left_image_flip = cv2.flip(left_image,1)
                left_angle = float(batch_sample[3])+0.3
                left_angle_flip = -left_angle 
                
                # Get the right images from the vehicle and flip it horizontally
                right_image = cv2.imread(name_right)
                right_image_flip = cv2.flip(right_image,1)
                right_angle = float(batch_sample[3])-0.3
                right_angle_flip = -right_angle
                
                # Add all images to the images array and all angles to the angles array
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)
                
                images.append(center_image_flip)
                angles.append(center_angle_flip)
                images.append(left_image_flip)
                angles.append(left_angle_flip)  
                images.append(right_image_flip)
                angles.append(right_angle_flip)                  

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)           
            #X_train = rgb2gray(X_train)

            yield sklearn.utils.shuffle(X_train, y_train)

# Setup the training and validation generators
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = get_model_cai()

# Load any previous models if they exist to continue training
if os.path.isfile("model.h5"):
    model = load_model("model.h5")

# Compile and train the model using the generator function
model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=3)

# Save the model to the disk
model.save('model.h5')