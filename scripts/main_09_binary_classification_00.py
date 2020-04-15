# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:24:43 2020

Binary classification

Model 2

CatDogNet-16 -Simple

Ensembling the classifiers:

@author: Ashiv Hans Dhondea
"""

# images to be resized to (image_dim) x (image_dim)
image_dim = 200;

name_brand_1 = 'Lexus';
name_brand_2 = 'Mercedes-Benz'
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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 

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
import json

# Tensorflow
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
# Split dataset into train, validate and test sets
def fn_data_split(examples, labels, train_frac, random_state=None):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    
    This function was taken from
    https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
    '''
    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"
    X_train, X_tmp, Y_train, Y_tmp = train_test_split(examples, labels, train_size=train_frac, random_state=random_state)
    X_val, X_test, Y_val, Y_test   = train_test_split(X_tmp, Y_tmp, train_size=0.5, random_state=random_state)
    return X_train, X_val, X_test,  Y_train, Y_val, Y_test
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

# To load the images in the dataset
dataset_x_files = [];

curated_file_list_cleaned = [None]*len(curated_file_list);
curated_file_list_classes = [None]*len(curated_file_list);
for k in range(len(curated_file_list)):
    # Trim the strings to get the file name
    filename = str(curated_file_list[k]);
    curated_file_list_cleaned[k] = filename[2:-2];
    curated_file_list_classes[k] = filename[2:].split("_")[0];
    # Location of image file in dataset
    file_loc = path_base+curated_file_list_classes[k]+"/"+curated_file_list_cleaned[k]; 
    # read image and resize using cv2
    image = cv2.imread(file_loc);
    dataset_x_files.append(cv2.resize(image,(image_dim, image_dim)));
    
    
# Find unique classes and their frequencies
curated_brands, curated_brands_freq = np.unique(curated_file_list_classes,return_counts=True);
print('There are %d %s images and %d %s images in the dataset.' % (curated_brands_freq[0],curated_brands[0],curated_brands_freq[1],curated_brands[1]));

# Load the data for both categories
dataset_y_labels =  curated_file_list_classes;
# --------------------------------------------------------------------------- #
# Pre-processing the examples' labels before getting into the learning phase
# change labels from strings to integers, e.g brand_1 -> 0, brand_2 -> 1
le = LabelEncoder();
y_dataset = le.fit_transform(dataset_y_labels);
x_dataset = np.array(dataset_x_files,np.float32);
   
# random state stored in a variable to be able to recover the exact order
random_state_chosen = 5;

# Creating a training set, test set and cross validation set from the dataset.
# 66% of the dataset is used for training
# 17% of the dataset is used for testing
# 17% of the dataset is used for cross validation
# Note that these are selected at random
train_size = 0.66;
val_size = 0.5*(1-train_size);
x_train, x_val, x_test, y_train,y_val,y_test = fn_data_split(x_dataset,y_dataset,train_size,random_state=random_state_chosen);

# Shuffle the training set, test set and validation set
x_train = shuffle(x_train, random_state=random_state_chosen);
y_train = shuffle(y_train, random_state=random_state_chosen);

x_test = shuffle(x_test, random_state=random_state_chosen);
y_test = shuffle(y_test, random_state=random_state_chosen);

x_val = shuffle(x_val, random_state=random_state_chosen);
y_val = shuffle(y_val, random_state=random_state_chosen);
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

model_name = '_model_2';
# --------------------------------------------------------------------------- #
start_t = timer();
# ---- train the model ----
batch_size_chosen = 32;
num_epochs = 100;

# ---- define data generator ----
datagen = ImageDataGenerator(rescale=1./255) 
datagen.fit(x_train)
datagen.fit(x_val)

datagen = ImageDataGenerator(rescale=1./255) # rescaling pixel values from [0,255] to [0,1]
datagen.fit(x_test)

# Set callback functions to early stop training and save the best model so far
callbacks_early_stopping = [EarlyStopping(monitor='val_loss', patience=20,restore_best_weights=True)];

history_of_model_1 = model_2.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size_chosen),
                    steps_per_epoch=len(x_train) / batch_size_chosen, epochs=num_epochs,callbacks=callbacks_early_stopping,
                    validation_data=datagen.flow(x_val, y_val, batch_size=batch_size_chosen),
                    validation_steps = len(x_val) / batch_size_chosen);
 
#stop the timer
end_t = timer()
chrono = end_t-start_t;
print("Elapsed time = {} seconds".format(chrono)); 
# --------------------------------------------------------------------------- #
# ---- save the model and the weights ----
model_2.save(full_name+model_name+'model.h5')
model_2.save_weights(full_name+model_name+'weights.h5')
print('Model saved\n')

# Get the dictionary containing each metric and the loss for each epoch
history_dict = history_of_model_1.history
# Save it under the form of a json file
json.dump(history_dict, open(full_name+model_name+'history.json', 'w'));

# Confusion Matrix and Classification Report
# Evaluate trained model on the test set                                       
test_results = model_2.evaluate(x_test, y_test, batch_size=batch_size_chosen);
print('test loss, test acc:', test_results);

test_results_df = pd.DataFrame(test_results).transpose();
with open(full_name+model_name+'test_results.tex', 'w') as texfile:
    texfile.write(test_results_df.to_latex())

y_pred_test_continuous =  model_2.predict(x_test,batch_size= batch_size_chosen);
np.save(full_name+model_name+'y_pred_test_continuous.npy',y_pred_test_continuous);
np.save(full_name+model_name+'y_test.npy',y_test);

y_pred_train_continuous =  model_2.predict(x_train,batch_size= batch_size_chosen);
np.save(full_name+model_name+'y_pred_train_continuous.npy',y_pred_train_continuous);
np.save(full_name+model_name+'y_train.npy',y_train);

target_names = [name_brand_1,name_brand_2];

y_pred = y_pred_test_continuous > 0.5;
report = classification_report(y_test, y_pred, target_names=target_names);
print("Classification report\n")
print(report)

report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(full_name+model_name+'classification_report.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())

confusion_matrix_test = confusion_matrix(y_test,y_pred);
confusion_matrix_test_df = pd.DataFrame(confusion_matrix_test).transpose();
with open(full_name+model_name+'confusion_matrix.tex', 'w') as texfile:
    texfile.write(confusion_matrix_test_df.to_latex());

print("\n Confusion matrix \n");
print(confusion_matrix_test)
# --------------------------------------------------------------------------- #
# Plot results
val_loss_min_arg = np.argmin(history_of_model_1.history['val_loss']);
val_loss_min = min(history_of_model_1.history['val_loss']);
val_loss_min_acc = max(history_of_model_1.history['acc'][val_loss_min_arg],history_of_model_1.history['val_acc'][val_loss_min_arg])

fig = plt.figure(4);
ax = fig.gca();
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
plt.rc('font',family='DejaVu Sans');
params = {'legend.fontsize': 8,
    'legend.handlelength': 2}
plt.rcParams.update(params)
plt.title(r"\textbf{Training and validation accuracy per epoch for %s v. %s}" %(name_brand_1,name_brand_2) ,fontsize=12)
plt.plot(history_of_model_1.history['acc'],color='mediumblue')
plt.plot(history_of_model_1.history['val_acc'],color='forestgreen')

#ax.axvline(val_loss_min_arg,color='maroon',linestyle='dashed');
plt.plot([val_loss_min_arg,val_loss_min_arg],[0,val_loss_min_acc],color='maroon',linestyle='dashed');

ax.set_xlabel(r'Epoch')
ax.set_ylabel(r'Accuracy'); 
xticks_range=np.arange(0,num_epochs,5,dtype=np.int64);
plt.xlim(0,num_epochs);
plt.xticks(xticks_range);
plt.ylim(0,1)
plt.legend([r'Train', r'Validation'], loc='upper left');
plt.annotate(r"\textbf{Batch size = %d}" %batch_size_chosen,(0,0.35) );
plt.annotate(r"\textbf{Validation set size = %.3f}" %val_size,(0,0.3) );
plt.annotate(r"\textbf{Validation accuracy = %.3f}" %history_of_model_1.history['val_acc'][val_loss_min_arg],(0,0.25) );
plt.annotate(r"\textbf{at minimum validation loss at epoch = %d}" %val_loss_min_arg, (0,0.2))
fig.savefig(full_name+model_name+'accuracy.pdf')


# summarize history for loss (binary cross-entropy)
fig = plt.figure(6);
ax = fig.gca();
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
plt.rc('font',family='DejaVu Sans');
params = {'legend.fontsize': 8,
    'legend.handlelength': 2}
plt.rcParams.update(params)
plt.title(r"\textbf{Training and validation loss per epoch: %s v. %s}" %(name_brand_1,name_brand_2) ,fontsize=12)
plt.plot(history_of_model_1.history['loss'],color='mediumblue')
plt.plot(history_of_model_1.history['val_loss'],color='forestgreen')

plt.plot([val_loss_min_arg,val_loss_min_arg],[0,val_loss_min],color='maroon',linestyle='dashed');
ax.set_xlabel(r'Epoch')
ax.set_ylabel(r'Binary cross-entropy'); 
xticks_range=np.arange(0,num_epochs,5,dtype=np.int64);
plt.xlim(0,num_epochs);
plt.xticks(xticks_range);
plt.ylim(0,1)
plt.annotate(r"\textbf{Batch size = %d}" %batch_size_chosen,(10,0.8) );
plt.annotate(r"\textbf{Validation set size = %.3f}" %val_size,(10,0.75) );
plt.annotate(r"\textbf{Minimum validation loss occurs at epoch = %d}" %val_loss_min_arg,(10,0.7) );
plt.annotate(r"\textbf{Minimum validation loss = %.6f}" %val_loss_min,(10,0.65))
plt.legend([r'Train', r'Validation'], loc='upper left')
plt.savefig(full_name+model_name+'loss.pdf')
# --------------------------------------------------------------------------- #
