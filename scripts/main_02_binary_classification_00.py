# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 01:20:41 2020

Binary classification task

Pre-processing:
    1. Exploratory Data Analysis
    2. Undersampling to achieve a balanced dataset

@author: Ashiv Hans Dhondea
"""
"""
Parameters used:
    image_dim = dimension of images resized
    name_brand_1 = name of first brand of car
    name_brand_2 = name of second brand of car

"""
# Select two brands for binary classification
name_brand_1 = 'Lexus';
name_brand_2 = 'Mercedes-Benz';

# --------------------------------------------------------------------------- #
# Import the necessary packages

# numpy for linear algebra, cv2 for image processing
#  glob and os to navigate directories
import numpy as np    
import random
import glob
import os
import sys      

# matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['DejaVu Sans']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params);

# pandas for excel sheet wrangling
import pandas as pd 
#import json

# close any open figure
plt.close('all');
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

# All files created by this script will be named according to this:
full_name = script_name+'_'+name_brand_1+'_'+name_brand_2+ "_undersampl";
# --------------------------------------------------------------------------- #
path_base = 'TCC_dataset/'
print('Available classes in the dataset are: ');
classes_list = os.listdir(path_base)
print(classes_list);
# --------------------------------------------------------------------------- #
# Load the dataset

# file type of interest
file_extension = "jpg";

classes_count = np.zeros([len(classes_list)],dtype=int);

# count how many examples there are for each class
for i in range(len(classes_list)):
    classes_count[i] = len(glob.glob1(path_base + classes_list[i]+"/","*."+file_extension));

classes_count_total = np.sum(classes_count);
print('Our dataset comprises of %d images.' %classes_count_total);

# calculate statistics of this dataset
classes_prob = classes_count*(1/np.sum(classes_count));
classes_mean = np.mean(classes_count);
classes_std = np.std(classes_count);

print("The mean number of examples is %.3f \n" %classes_mean);
print("The standard deviation is %.3f examples. \n" %classes_std);

# Choose brands for classification
chosen_classes = [name_brand_1,name_brand_2];
print('We will classify images between the following classes:');
print(chosen_classes);

# Count number of examples for each class
chosen_classes_num = np.zeros([len(chosen_classes)],dtype=int);
for i in range(len(chosen_classes)):
    chosen_classes_num[i] = classes_count[classes_list.index(chosen_classes[i])];

chosen_classes_total = np.sum(chosen_classes_num);
print('This subset consists of %d images.' %chosen_classes_total);
# --------------------------------------------------------------------------- #
fig = plt.figure(1);
pos = np.arange(len(classes_list));

color_list = ['limegreen','indianred','teal','darkorange','cornflowerblue','lightsalmon'];

for index in pos:
    plt.bar(index,classes_count[index],color=color_list[index],label=r"%.3f" %(classes_prob[index]));

plt.xticks(pos,classes_list);
plt.title(r"\textbf{Distribution of classes in the} \textit{TCC dataset}",fontsize=12)
plt.xlabel(r"\textbf{Classes}")
plt.ylabel(r"\textbf{Count}")
plt.legend(loc='upper left');
#plt.savefig(full_name+'_full_dataset.png');
plt.savefig(full_name+'full_dataset.pdf');
plt.show();
# --------------------------------------------------------------------------- #
# Find the least represented class and undersample the other class
smallest_count_chosen = np.min(chosen_classes_num);
smallest_count_chosen_index = np.argmin(chosen_classes_num);
smallest_count_chosen_id = chosen_classes[smallest_count_chosen_index];
print('The least represented class is %s which has %d examples.' %(smallest_count_chosen_id,smallest_count_chosen));

print('We will undersample the other class so that we end up with a balanced dataset')

# Create list of file names for each class to undersample
# Choose randomly in this list to obtain the required number of examples

overall_files_list = [];

for i in range(0,len(chosen_classes)):
    files_list = [];
    for file in glob.glob(path_base+"/"+chosen_classes[i]+"/*."+file_extension):
        index_for_filename = file.index('\\');
        files_list.append(file[index_for_filename+1:]);
    random.shuffle(files_list);
    overall_files_list.extend(files_list[:smallest_count_chosen]);
            
df_list = pd.DataFrame(overall_files_list);
df_list.to_excel(full_name+'.xlsx', engine='xlsxwriter')
print('Examples per class:')  
print(len(overall_files_list)/len(chosen_classes));
# --------------------------------------------------------------------------- #
# Load excel sheet and verify the distribution of classes
# Read the excel file and pick out the images which are relevant to this script
worksheet_name = 'Sheet1';
list_file = full_name+'.xlsx';
data_frames = pd.read_excel(list_file, sheet_name=worksheet_name);
curated_file_list = np.asarray(data_frames.values.tolist());

curated_file_list_cleaned = [None]*len(curated_file_list);
curated_file_list_classes = [None]*len(curated_file_list);
for k in range(len(curated_file_list)):
    filename = str(curated_file_list[k]);
    curated_file_list_cleaned[k] = filename[2:-2];
    curated_file_list_classes[k] = filename[2:].split("_")[0];

# Find unique classes and their frequencies
curated_brands, curated_brands_freq = np.unique(curated_file_list_classes,return_counts=True);

# Compute stats for the undersampled dataset
curated_brands_prob = np.asarray(curated_brands_freq,dtype=np.float64)*(1/np.sum(np.asarray(curated_brands_freq,dtype=np.float64)));
curated_brands_mean = np.mean(np.asarray(curated_brands_freq,dtype=np.float64));
curated_brands_std = np.std(np.asarray(curated_brands_freq,dtype=np.float64));

print('For the undersampled dataset:')
print("The mean number of examples is %.3f " %curated_brands_mean);
print("The standard deviation is %.3f examples." %curated_brands_std);
# --------------------------------------------------------------------------- #
# Plot the selected dataset (after undersampling)
fig = plt.figure(2);
pos = np.arange(len(curated_brands));

color_list = ['limegreen','indianred','teal','darkorange','cornflowerblue','lightsalmon'];

for index in pos:
    plt.bar(index,curated_brands_freq[index],color=color_list[index],edgecolor='dimgray',label=r"%.3f" %(curated_brands_prob[index]));

plt.xticks(pos,curated_brands);
plt.title(r"\textbf{Distribution of classes in the curated} \textit{TCC dataset}",fontsize=12)
plt.xlabel(r"\textbf{Classes}")
plt.ylabel(r"\textbf{Count}")
plt.legend(loc='upper left');
#plt.savefig(full_name+'_balanced_dataset.png');
plt.savefig(full_name+'balanced_dataset.pdf');
plt.show();
# --------------------------------------------------------------------------- #
