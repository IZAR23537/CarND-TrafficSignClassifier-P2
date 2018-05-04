# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/IZAR23537/CarND-TrafficSignClassifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Visualization of the dataset.

Here is a few random image of the data set.

[alt text][./write_up_images/image1]

### Design and Test a Model Architecture

#### 1. Preprocessed the image data. 

As a first step, I decided to convert the images to grayscale in order to reduce the image depth.

Next, I normalized the image data by divided the image values by 127.5 and subtracting it by 1.

#### 2. Describing the final model architecture.

My final model consisted of the following layers:

| Layer 				 |     Description								| 
|:----------------------:|:--------------------------------------------:| 
| Input					 | 32x32x1 Gray image 							| 
| Convolution Layer 1  	 | 1x1 stride, Valid padding, outputs 28x28x16 	|
| RELU					 |												|
| Max pooling			 | 2x2 stride,  outputs 14x14x16 				|
| Convolution Layer 2    | 1x1 stride, Valid padding, outputs 10x10x32 	|
| Fatten				 | outputs 800 									|
| Fully connected Layer 3| outputs 240 									|
| RELU					 | 												|
| Fully connected Layer 4| outputs 84 									|
| RELU					 | 												|
| Fully connected Layer 5| outputs 43 									|
 

#### 3. Training, validating and testing the model. 

I started the traffic sign validation with LeNet as a base model which was provided by Udacity.

To train the model, I started to initialize the model with the original images, 10 epochs, 0.001 learning rate and 128 batch size.

The model at that state produced a low validation accuracy, therefore I preprocessed the images (gray scaled and normalized).

Because of the gray scaling I had to change the model architecture. I changed the input layer and I also changed the convolutional layers by making it deeper and the fully-connected layers output size larger as well.

After the model change, the validation accuracy is started to rise above 90%. 

I changed the number of epochs from 10 to 25 and the accuracy is risen above 93%.

My final model results were:
    Training set accuracy: 1.0
    Validation set accuracy: 0.942
    Test set accuracy: 0.938
    

### Test a Model on New Images

#### 1. Testing five German traffic signs found on the web

Here are five German traffic signs that I found on the web:

![alt text][https://write_up_images/image2.jpg.jpg]

These new images have different sizes and contains different traffic sign.

After preprocessing the images:

[alt text][./write_up_images/image3]

As a next step I evaluated the new images. 

Four out of the five images were classified correctly. It means that the network recognized 80% of the new images.

#### 2. Top 5 softmax probabilities on new images

Here are the results of the prediction:

[alt text][./write_up_images/image4]