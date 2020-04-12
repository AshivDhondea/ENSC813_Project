# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 00:58:58 2020

The Car Connection Dataset

Creates an excel file listing the names of images employed in this project 
for reproducibility purposes.

Use the excel file to select files from the original raw dataset to be included
in this project.

@author: Ashiv Hans Dhondea
"""
# --------------------------------------------------------------------------- #
# Import the necessary packages

# numpy for linear algebra, cv2 for image processing
#  glob and os to navigate directories
#import numpy as np    
import xlsxwriter
import glob
import os
import sys      

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
full_name = script_name+'_';
# --------------------------------------------------------------------------- #
path_base = 'TCC_dataset/'
print('Available classes in the dataset are: ');
classes_list = os.listdir(path_base)
print(classes_list);
# --------------------------------------------------------------------------- #
# Load the dataset
# file type of interest
file_extension = "jpg";

# Create list of file names and store them in an excel worksheet for later 
# reference
workbook = xlsxwriter.Workbook(full_name+"files_list.xlsx");

for i in range(0,len(classes_list)):
    worksheet_name = classes_list[i];
    worksheet = workbook.add_worksheet(worksheet_name);
    print('Processing images for class %s' %classes_list[i]);
    
    # Start from entry 0,0 in worksheet
    row = 0;
    column = 0;
      
    for file in glob.glob(path_base+"/"+classes_list[i]+"/*."+file_extension):
        index_for_filename = file.index('\\');
        worksheet.write(row, column, file[index_for_filename+1:]); 
        row += 1; # Increment to get to next row in worksheet.
    
workbook.close(); 
# --------------------------------------------------------------------------- #
print('Excel file containing relevant file names has been created.')
