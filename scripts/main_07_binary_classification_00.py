# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:44:36 2020

Binary classification

Benchmarking the ConvNets used

Use the best accuracy results on the test set from k-fold cross validation as a
measure of performance.

@author: Ashiv Hans Dhondea
"""
import numpy as np
import pandas as pd
import os
import sys

# --------------------------------------------------------------------------- #
# Classification task number 3 is Honda v. Toyota -> task = 2 

names_1 = ['Audi','Lexus','Honda'];
names_2 = ['BMW','Mercedes-Benz','Toyota'];

"""
Model 1 - Simple CNN implemented in
'main_03_binary_classification_00.py'

Model 2 - implemented in
'main_04_binary_classification_00.py'

Model 3 - implemented in 
'main_05_binary_classification_00.py'

Model 4 - implemented in
'main_06_binary_classification_00.py'
"""

script_names = ['main_03_binary_classification_00','main_04_binary_classification_00','main_05_binary_classification_00','main_06_binary_classification_00'];

results_acc = np.zeros([len(script_names),len(names_1)],dtype=np.float64);
#results_loss = np.zeros([len(script_names),len(names_1)],dtype=np.float64);

for script in range(len(script_names)):
    for task in range(len(names_1)):
        full_name = script_names[script]+'_'+names_1[task]+'_'+names_2[task]+'_.npy';
        results = np.load(full_name);
        results_acc[script,task] = np.max(results[1,:]);

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

full_name = script_name+'_';
# --------------------------------------------------------------------------- #
# save as latex file.
results_acc_df = pd.DataFrame(results_acc)
with open(full_name+'kfold_accuracy.tex', 'w') as texfile:
    texfile.write(results_acc_df.to_latex());
