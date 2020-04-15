# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 02:40:09 2020

Multiclass classification

Model 2

Ensembling the classifiers

@author: Ashiv Hans Dhondea
"""
# images to be resized to (image_dim) x (image_dim)
image_dim = 128;
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

from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical

# matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['DejaVu Sans']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params);

plt.close('all');
import seaborn as sn

# pandas for excel sheet wrangling
import pandas as pd 
import json


from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

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
# Sort out utilities for file naming
# get the name of this script
file_name =  os.path.basename(sys.argv[0]);

if file_name[-3:] == '.py':
    script_name = file_name[:-3];
elif file_name[-3:] == '.ipynb':
    script_name = file_name[:-6];
else:
      script_name = 'main_xx';  

# --------------------------------------------------------------------------- #
# Load the dataset
file_extension = "jpg";
path_base = 'TCC_dataset/'
# --------------------------------------------------------------------------- #
# Load excel sheet and verify the distribution of classes
# Read the excel file and pick out the images which are relevant to this script
worksheet_name = 'Sheet1';
list_file = 'main_19_multiclass_classification_00_undersampl.xlsx';
data_frames = pd.read_excel(list_file, sheet_name=worksheet_name);
curated_file_list = data_frames.values.tolist();

dataset_y_labels = [None]*len(curated_file_list);
dataset_x_files = [];

for k in range(len(curated_file_list)):
    filename = str(curated_file_list[k]);   
    dataset_y_labels[k] = filename[2:].split("_")[0];
    image = cv2.imread(path_base+filename[2:].split("_")[0]+'/'+filename[2:-2]);
    dataset_x_files.append(cv2.resize(image,(image_dim,image_dim)));
    
x_dataset = np.array(dataset_x_files,np.float32);

curated_brands, curated_brands_freq = np.unique(np.asarray(dataset_y_labels),return_counts=True);

le = LabelEncoder();
datay = le.fit_transform(dataset_y_labels);
y_dataset = to_categorical(datay);

output_nodes = len(curated_brands);

# --------------------------------------------------------------------------- #
# From the dataset, we create a set for testing (performance evaluation) and a 
# set for training and validation.

# random state stored in a variable to be able to recover the exact order
random_state_chosen = 13;

# Creating a training set, test set and cross validation set from the dataset.
# 66% of the dataset is used for training
# 17% of the dataset is used for testing
# 17% of the dataset is used for cross validation
# Note that these are selected at random
train_size = 0.66;
val_size = 0.5*(1-train_size);
test_size = val_size;
x_train, x_val, x_test, y_train,y_val,y_test = fn_data_split(x_dataset,y_dataset,train_size,random_state=random_state_chosen);

# Shuffle the training set, test set and validation set
x_train = shuffle(x_train, random_state=random_state_chosen);
y_train = shuffle(y_train, random_state=random_state_chosen);

x_test = shuffle(x_test, random_state=random_state_chosen);
y_test = shuffle(y_test, random_state=random_state_chosen);

x_val = shuffle(x_val, random_state=random_state_chosen);
y_val = shuffle(y_val, random_state=random_state_chosen);

y_val_cat = np.argmax(y_val,axis=1);
y_test_cat = np.argmax(y_test,axis=1);
y_train_cat = np.argmax(y_train,axis=1);

unique, counts = np.unique(y_val_cat, return_counts=True);
y_val_count_dic = dict(zip(unique, counts));

unique, counts = np.unique(y_test_cat, return_counts=True);
y_test_count_dic = dict(zip(unique, counts));

unique, counts = np.unique(y_train_cat, return_counts=True);
y_train_count_dic = dict(zip(unique, counts));

# Number of output nodes is the number of unique classes in the dataset
output_nodes = len(unique);
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

full_name = script_name;
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
model_2.add(Dense(output_nodes))
model_2.add(Activation('softmax'))

# learning rate
learning_rate = 0.00002;
opt = Adam(lr = learning_rate);

model_2.compile(loss='binary_crossentropy',optimizer=opt,metrics=['categorical_accuracy']);

model_name = '_model_2';
# --------------------------------------------------------------------------- #
start_t = timer();
# ---- train the model ----
batch_size_chosen = 32;
num_epochs = 200;

# ---- define data generator ----
datagen = ImageDataGenerator(rescale=1./255) 
datagen.fit(x_train)
datagen.fit(x_val)

datagen = ImageDataGenerator(rescale=1./255) # rescaling pixel values from [0,255] to [0,1]
datagen.fit(x_test)

# Set callback functions to early stop training and save the best model so far
callbacks_early_stopping = [EarlyStopping(monitor='val_loss', patience=20,restore_best_weights=True)];

history_of_model_2 = model_2.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size_chosen),
                    steps_per_epoch=len(x_train) / batch_size_chosen, epochs=num_epochs,callbacks=callbacks_early_stopping,
                    validation_data=datagen.flow(x_val, y_val, batch_size=batch_size_chosen),
                    validation_steps = len(x_val) / batch_size_chosen);
 
#stop the timer
end_t = timer()
chrono = end_t-start_t;
print("Elapsed time = {} seconds".format(chrono)); 
## --------------------------------------------------------------------------- #
# ---- save the model and the weights ----
model_2.save(full_name+'_model.h5')
model_2.save_weights(full_name+'_weights.h5')
print('Model saved\n')

# Get the dictionary containing each metric and the loss for each epoch
history_dict = history_of_model_2.history
## Save it under the form of a json file
json.dump(history_dict, open(full_name+'_history.json', 'w'));

# Evaluate trained model on the test set                                       
test_results = model_2.evaluate(x_test, y_test, batch_size=batch_size_chosen);
print('test loss, test acc:', test_results);

y_pred_test_continuous =  model_2.predict(x_test,batch_size= batch_size_chosen);
y_pred = np.argmax(y_pred_test_continuous,axis=1);
y_test_enc = np.argmax(y_test,axis=1);
y_test_compare = np.equal(y_test_enc,y_pred);
y_pred_test_accuracy = np.count_nonzero(y_test_compare)/np.shape(y_test_compare)[0];

print('Categorical accuracy: %.6f' %y_pred_test_accuracy)
# --------------------------------------------------------------------------- #

test_results_df = pd.DataFrame(test_results).transpose();
with open(full_name+'test_results.tex', 'w') as texfile:
    texfile.write(test_results_df.to_latex())
    
np.save(full_name+'_y_pred_test_continuous.npy',y_pred_test_continuous);
np.save(full_name+'_y_test.npy',y_test);

y_pred_train_continuous =  model_2.predict(x_train,batch_size= batch_size_chosen);
np.save(full_name+'_y_pred_train_continuous.npy',y_pred_train_continuous);
np.save(full_name+'_y_train.npy',y_train);
# --------------------------------------------------------------------------- #
confusion_matrix = confusion_matrix(y_test_enc, y_pred)

confusion_matrix_df = pd.DataFrame(confusion_matrix);
with open(full_name+'_confusion_matrix.tex', 'w') as texfile:
    texfile.write(confusion_matrix_df.to_latex())
    
df_cm = pd.DataFrame(confusion_matrix, curated_brands, curated_brands)
df_cm.index.name = r'\textbf{Actual}'
df_cm.columns.name = r'\textbf{Predicted}'
plt.figure(figsize=(10,7))
#sn.set(font_scale=1.4) # for label size
plt.title(r"\textbf{Confusion matrix on test set with Model 2}",fontsize=12);
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
plt.savefig(full_name+'_confusion_matrix.pdf');
plt.show()
# --------------------------------------------------------------------------- #
# Plot results
val_loss_min_arg = np.argmin(history_of_model_2.history['val_loss']);
val_loss_min = min(history_of_model_2.history['val_loss']);
val_loss_min_acc = max(history_of_model_2.history['categorical_accuracy'][val_loss_min_arg],history_of_model_2.history['val_categorical_accuracy'][val_loss_min_arg])

fig = plt.figure(4);
ax = fig.gca();
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
plt.rc('font',family='DejaVu Sans');
params = {'legend.fontsize': 8,
    'legend.handlelength': 2}
plt.rcParams.update(params)
plt.title(r"\textbf{Training and validation accuracy per epoch}" ,fontsize=12)
plt.plot(history_of_model_2.history['categorical_accuracy'],color='mediumblue')
plt.plot(history_of_model_2.history['val_categorical_accuracy'],color='forestgreen')
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
plt.annotate(r"\textbf{Validation accuracy = %.3f}" %history_of_model_2.history['val_categorical_accuracy'][val_loss_min_arg],(0,0.25) );
plt.annotate(r"\textbf{at minimum validation loss at epoch = %d}" %val_loss_min_arg, (0,0.2))
fig.savefig(full_name+'_accuracy.pdf')

# summarize history for loss (binary cross-entropy)
fig = plt.figure(6);
ax = fig.gca();
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
plt.rc('font',family='DejaVu Sans');
params = {'legend.fontsize': 8,
    'legend.handlelength': 2}
plt.rcParams.update(params)
plt.title(r"\textbf{Training and validation loss per epoch}" ,fontsize=12)
plt.plot(history_of_model_2.history['loss'],color='mediumblue')
plt.plot(history_of_model_2.history['val_loss'],color='forestgreen')

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
plt.savefig(full_name+'_loss.pdf')

# --------------------------------------------------------------------------- #
