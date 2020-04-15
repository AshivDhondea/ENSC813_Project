# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:34:32 2020

Model 2

CatDogNet-16 -Simple

Classifying images between two brands
e.g Lexus vs. Mercedes-Benz

Ref: The CNN used is CatDognet16-Simplified


@author: Ashiv Hans Dhondea
"""
"""
Parameters used:
    image_dim = dimension of images resized
    name_brand_1 = name of first brand of car
    name_brand_2 = name of second brand of car

"""
# images to be resized to (image_dim) x (image_dim)
image_dim = 200;

name_brand_1 = 'Lexus';
name_brand_2 = 'Mercedes-Benz';
# --------------------------------------------------------------------------- #
# Import the necessary packages

# numpy for linear algebra, cv2 for image processing
#  glob and os to navigate directories
import numpy as np    
import cv2 
import os
import sys      

# sklearn 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle # for shuffling the set
from sklearn.model_selection import KFold

from keras.callbacks import EarlyStopping

# matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['DejaVu Sans']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params);

# pandas for excel sheet wrangling
import pandas as pd 

## Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.optimizers import Adam

import tensorflow as tf
if tf.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()));
else:
   print("Please install GPU version of TF");

# to measure elapsed time
from timeit import default_timer as timer
# --------------------------------------------------------------------------- #
path_base = 'TCC_dataset/'
print('Available categories in the dataset are: ');
print(os.listdir(path_base));
# --------------------------------------------------------------------------- #
# Load the dataset
file_extension = "jpg";

# Load excel sheet and verify the distribution of classes
# Read the excel file and pick out the images which are relevant to this script
worksheet_name = 'Sheet1';
list_file = 'main_02_binary_classification_00_'+name_brand_1+'_'+name_brand_2+'_undersampl.xlsx';
data_frames = pd.read_excel(list_file, sheet_name=worksheet_name);
curated_file_list = np.asarray(data_frames.values.tolist());

dataset_x_files = [];

curated_file_list_cleaned = [None]*len(curated_file_list);
curated_file_list_classes = [None]*len(curated_file_list);
for k in range(len(curated_file_list)):
    filename = str(curated_file_list[k]);
    curated_file_list_cleaned[k] = filename[2:-2];
    curated_file_list_classes[k] = filename[2:].split("_")[0];
    file_loc = path_base+curated_file_list_classes[k]+"/"+curated_file_list_cleaned[k]; 
    image = cv2.imread(file_loc);
    dataset_x_files.append(cv2.resize(image,(image_dim, image_dim)));
    
    
# Find unique classes and their frequencies
curated_brands, curated_brands_freq = np.unique(curated_file_list_classes,return_counts=True);
print('There are %d %s images and %d %s images in the dataset.' % (curated_brands_freq[0],curated_brands[0],curated_brands_freq[1],curated_brands[1]));

# Load the data for both categories
dataset_y_labels =  curated_file_list_classes;

# change labels from strings to integers, e.g brand_1 -> 0, brand_2 -> 1
le = LabelEncoder();
y_dataset = le.fit_transform(dataset_y_labels);
x_dataset = np.array(dataset_x_files,np.float32);
   
# From the dataset, we create a set for testing (performance evaluation) and a 
# set for training and validation.
# We choose a split of 80% for training + validation
# Therefore, we hold 20% for testing.
# We select these at random.

# random state stored in a variable to be able to recover the exact order
random_state_chosen = 42;
# Splitting the data set into a training set and a test set according to test_size
test_size = 0.2;
x_trainval, x_test, y_trainval, y_test = train_test_split(x_dataset,y_dataset,test_size=test_size,random_state=random_state_chosen)

# x_tranval and y_trainval are used for training the CNN and evaluating it.
# The best model parameters are chosen from the best validation error obtained
# from the validation set.

# Shuffle the training and test set
x_trainval = shuffle(x_trainval, random_state=random_state_chosen);
y_trainval = shuffle(y_trainval, random_state=random_state_chosen);

x_test = shuffle(x_test, random_state=random_state_chosen);
y_test = shuffle(y_test, random_state=random_state_chosen);
# --------------------------------------------------------------------------- #
# Sort out utilities for file naming
# get the name of this script
file_name =  os.path.basename(sys.argv[0]);

if file_name[-3:] == '.py':
    script_name = file_name[:-3];
elif file_name[-3:] == '.ipynb':
    script_name = file_name[:-6];
else:
      script_name = 'main_xx';  

full_name = script_name+'_'+name_brand_1+'_'+name_brand_2+'_';
# --------------------------------------------------------------------------- #
# Display the size of the train+val and test sets

#fig = plt.figure(1);
#plt.title(r"\textbf{Split: train-validate set (%.3f) versus test set (%.3f)}" %(1-test_size,test_size) ,fontsize=12)
#n_groups = 3;
#index = np.arange(n_groups);
#bar_width = 0.35
#opacity = 0.8
#num_brand_1_trainval = np.count_nonzero(y_trainval==0);
#num_brand_1_test = np.count_nonzero(y_test==0);
#name_brand_1_split = np.array([curated_brands_freq[0],num_brand_1_trainval,num_brand_1_test]);
#rects1 = plt.bar(index, name_brand_1_split, bar_width,alpha=opacity,color='seagreen',label=r'%s' %name_brand_1)
#name_brand_2_split = np.array([curated_brands_freq[1],len(y_trainval) - num_brand_1_trainval,len(y_test)-num_brand_1_test]);
#rects2 = plt.bar(index + bar_width, name_brand_2_split, bar_width,alpha=opacity,color='royalblue',label=r'%s' %name_brand_2)
#plt.xticks(index + 0.5*bar_width, (r'Dataset', r'TrainVal set',r'Test set'))
#plt.xlabel(r"\textbf{Set}")
#plt.ylabel(r"\textbf{Count}")
#plt.legend(loc='upper right');
#plt.tight_layout()
#plt.savefig(full_name+'split.png');
#plt.savefig(full_name+'split.pdf');
#plt.show();

# --------------------------------------------------------------------------- #
# Define the model architecture
# First Layer --------------------------------------------------------------  # 
model_2 = Sequential()
model_2.add(Conv2D(32, (3, 3), input_shape=(image_dim, image_dim, 3)))
model_2.add(Activation('relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
# Second Layer -------------------------------------------------------------  #
model_2.add(Conv2D(64, (3, 3)))
model_2.add(Activation('relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
# Third Layer --------------------------------------------------------------  # 
model_2.add(Conv2D(128, (3, 3)))
model_2.add(Activation('relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
# Fourth Layer --------------------------------------------------------------  # 
model_2.add(Conv2D(256, (3, 3)))
model_2.add(Activation('relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))
# ---- classification ------------------------------------------------------- #
model_2.add(Flatten())  # this produces a 1D feature vector
model_2.add(Dense(256))
model_2.add(Activation('relu'))
model_2.add(Dropout(0.5))

model_2.add(Dense(1))
model_2.add(Activation('sigmoid'))

# learning rate
learning_rate = 0.00005;
opt = Adam(lr = learning_rate);

model_2.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy']);
# --------------------------------------------------------------------------- #
## k-fold cross-validation
# ---- define data generator ----
datagen = ImageDataGenerator(rescale=1./255) # rescaling pixel values from [0,255] to [0,1]
datagen.fit(x_trainval)

datagen = ImageDataGenerator(rescale=1./255) # rescaling pixel values from [0,255] to [0,1]
datagen.fit(x_test)

# Set callback functions to early stop training and save the best model so far
callbacks_early_stopping = [EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)];

# start the timer
start_t = timer()

# train the model with the following batch_size and max number of epochs
batch_size_chosen = 32;
num_epochs = 100;

num_folds = 10;
# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

test_results = np.zeros([2,num_folds],dtype=np.float64);
target_names = [name_brand_1,name_brand_2];

# K-fold Cross Validation model evaluation
fold_no = 1
for train, val in kfold.split(x_trainval,y_trainval):

    # Generate a print
    print('-----------------------------------------------------------------');
    print('Training for fold %d \n' %fold_no)
    # --------------------------------------------------------------------------- #
    history_of_model_1 = model_2.fit_generator(datagen.flow(x_trainval[train], y_trainval[train], batch_size=batch_size_chosen),
                    steps_per_epoch=len(x_trainval[train]) / batch_size_chosen, epochs=num_epochs,callbacks=callbacks_early_stopping,
                    validation_data=datagen.flow(x_trainval[val], y_trainval[val], batch_size=batch_size_chosen),
                    validation_steps = len(x_trainval[val]) / batch_size_chosen);
                                                               
                                                                                                                           
    # Evaluate trained model on the test set                                       
    test_results[:,fold_no-1] = model_2.evaluate(x_test, y_test, batch_size=batch_size_chosen);
    
    # Increase fold number
    fold_no = fold_no + 1

#stop the timer
end_t = timer()
chrono = end_t-start_t;
print("Elapsed time = {} seconds".format(chrono));   
# --------------------------------------------------------------------------- #
np.save(full_name+'.npy',test_results);
print('The test results are:')
print(test_results[1,:])
# -------------------------------------------------------------------------- #

