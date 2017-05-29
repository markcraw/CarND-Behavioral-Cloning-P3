# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2016_12_01_13_30_48_287.jpg "Center Image"
[image2]: ./examples/left_2016_12_01_13_38_58_853.jpg "Recovery Left Image"
[image3]: ./examples/right_2016_12_01_13_36_22_036.jpg "Recovery RightImage"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 8x8 & 5x5 filter sizes and depths between 16 and 64 (model.py lines 39-62) 

The model includes ELU layers to introduce nonlinearity (code lines 46, 48, 54 & 57), the data is normalized in the model using a Keras lambda layer (code line 41) and the imputs are cropped using a Keras layer (code line 44). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 53 & 56). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 120 & 121). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 60).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.  I also mirrored the data for the center, left and right images along with mirroring their corresponding steering angle values.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to test a CNN model by evaluationg its performance using the Udacity simulation.

My first step was to use a convolution neural network model similar to the published model by CommaAI for their image/steering project. I thought this model might be appropriate because it proved success for the company's research efforts.  My first efforts just used the center image data from the vehicle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a higher mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that by increasing the effect of dropout.

Then I began to add more training data.  I added the left and right images.  The model performed better during autonomous testing, but was "pulling" the vehicle to one side of the road.  Next, I decided to mirror the data to adjust for the bias in turning direction for the track.  During the testing after the additon and training with  the new data, the overall performance of the model was greatly improved. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 39-62) consisted of a convolution neural network with the following layers and layer sizes 

1)  Convolution - filter 8x8, depth 16
2)  ELU - Exponential Linear Unit
3)  Convolution - filter 5x5, depth 32
4)  ELU - Exponential Linear Unit
5)  Convolution - filter 5x5, depth 64
6)  Flatten
7)  Dropout - keep 0.2
8)  ELU - Exponential Linear Unit
9)  Dense Fully Connected  - units 512
10) Dropout - keep 0.5
11) ELU - Exponential Linear Unit
12) Dense Fully Connected  - units 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the default data fromt he project. My model architecture was good enough that I did not need to record any additional data. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image2]
![alt text][image3]


To augment the data sat, I also flipped images and angles thinking that this would assit in reducing the bias in the data from the high appearance of left turns. I did not generate this images to disk and created them on the fly during the loading of the data sets.



Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
