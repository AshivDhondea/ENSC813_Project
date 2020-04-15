# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 03:57:32 2020

Binary classification

Ensembling the classifiers 
using normalized correlation combination

Saving the results in tex files for easy importing in the report

@author: Ashiv Hans Dhondea
"""
# --------------------------------------------------------------------------- #

# Import the necessary packages
# numpy for linear algebra, cv2 for image processing
#  glob and os to navigate directories
import numpy as np    
import pandas as pd

import os
import sys      

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 

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
# Classification task number 3 is Honda v. Toyota -> task = 2 

names_1 = ['Audi','Lexus','Honda'];
names_2 = ['BMW','Mercedes-Benz','Toyota'];

"""
Model 1 - Simple CNN implemented in
'main_08_binary_classification_00.py'

Model 2 - implemented in
'main_09_binary_classification_00.py'

Model 3 - implemented in 
'main_10_binary_classification_00.py'

Model 4 - implemented in
'main_11_binary_classification_00.py'
"""

script_names = ['main_08_binary_classification_00','main_09_binary_classification_00','main_10_binary_classification_00','main_11_binary_classification_00'];
model_names = ['__model_1','__model_2','__model_3','__model_4'];

num_comparisons = len(names_2);
num_methods = len(model_names);
# --------------------------------------------------------------------------- #
# Audi vs BMW classification
comparison = 0;
print('Classifying %s vs. %s' %(names_1[comparison],names_2[comparison]))

model = 0;
y_test_1 = np.load(script_names[0]+'_'+names_1[comparison]+'_'+names_2[comparison]+model_names[model]+'y_test.npy')
y_pred_test_continuous_arr_1 = np.zeros([num_methods,len(y_test_1)],dtype=np.float64);

y_pred_train_continuous_1 = np.load(script_names[0]+'_'+names_1[comparison]+'_'+names_2[comparison]+model_names[model]+'y_pred_train_continuous.npy')
y_pred_train_continuous_arr_1 = np.zeros([num_methods,len(y_pred_train_continuous_1)],dtype=np.float64);

y_train_1 = np.load(script_names[0]+'_'+names_1[comparison]+'_'+names_2[comparison]+model_names[model]+'y_train.npy')

for script in range(0,len(script_names)):    
    npy_file_name = script_names[script]+'_'+names_1[comparison]+'_'+names_2[comparison]+model_names[script];
    y_pred_test_continuous_arr_1[script,:] = np.ravel(np.load(npy_file_name+'y_pred_test_continuous.npy'));
    y_pred_train_continuous_arr_1[script,:] = np.ravel(np.load(npy_file_name+'y_pred_train_continuous.npy'));

r_y_pred_train_1 = np.dot(y_train_1.astype(np.float32), y_pred_train_continuous_arr_1[0,:])/len(y_pred_train_continuous_arr_1[0,:]);
r_y_pred_train_2 = np.dot(y_train_1.astype(np.float32), y_pred_train_continuous_arr_1[1,:])/len(y_pred_train_continuous_arr_1[1,:]);
r_y_pred_train_3 = np.dot(y_train_1.astype(np.float32), y_pred_train_continuous_arr_1[2,:])/len(y_pred_train_continuous_arr_1[2,:]);
r_y_pred_train_4 = np.dot(y_train_1.astype(np.float32), y_pred_train_continuous_arr_1[3,:])/len(y_pred_train_continuous_arr_1[3,:]);

# Correlation vector
r_mat = np.array([[r_y_pred_train_1],[r_y_pred_train_2],[r_y_pred_train_3],[r_y_pred_train_4]])

# normalized correlation vector
a_weights = r_mat/np.sum(r_mat);

# binarize predictions
y_pred_test_1_b = (y_pred_test_continuous_arr_1[0,:] > 0.5).astype(int)
y_pred_test_2_b = (y_pred_test_continuous_arr_1[1,:] > 0.5).astype(int)
y_pred_test_3_b = (y_pred_test_continuous_arr_1[2,:] > 0.5).astype(int)
y_pred_test_4_b = (y_pred_test_continuous_arr_1[3,:] > 0.5).astype(int)

# model average
prob_arr = np.transpose(y_pred_test_continuous_arr_1);
y_pred_test_continuous_mean = np.dot(prob_arr,a_weights);

y_pred_test_mean_b = (y_pred_test_continuous_mean > 0.5).astype(int);

# compute accuracies
y_pred_test_1_b_acc = np.mean(y_pred_test_1_b.ravel() == y_test_1)
y_pred_test_2_b_acc = np.mean(y_pred_test_2_b.ravel() == y_test_1)
y_pred_test_3_b_acc = np.mean(y_pred_test_3_b.ravel() == y_test_1)
y_pred_test_4_b_acc = np.mean(y_pred_test_4_b.ravel() == y_test_1)
y_pred_test_mean_acc = np.mean(y_pred_test_mean_b.ravel() == y_test_1)

columns_names = ['Accuracy'];
method_names = ['Model 1','Model 2','Model 3','Model 4','Ensemble'];

data_results = np.array([y_pred_test_1_b_acc,y_pred_test_2_b_acc,y_pred_test_3_b_acc,y_pred_test_4_b_acc,y_pred_test_mean_acc])
df_table_acc = pd.DataFrame(data = data_results,index=method_names, columns=columns_names)

print('Binary classification task %d' %(comparison+1))
print(df_table_acc)

results_table_name = full_name+'_'+names_1[comparison]+'_'+names_2[comparison];

with open(results_table_name+'_accuracy.tex', 'w') as texfile:
    texfile.write(df_table_acc.to_latex())

report = classification_report(y_test_1,y_pred_test_1_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[0]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())
    
report = classification_report(y_test_1,y_pred_test_2_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[1]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())    
    
report = classification_report(y_test_1,y_pred_test_3_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[2]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())
    
report = classification_report(y_test_1,y_pred_test_4_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[3]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())

report = classification_report(y_test_1,y_pred_test_mean_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[4]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex()) 
    
confusion_matrix_test = confusion_matrix(y_test_1,y_pred_test_mean_b);
confusion_matrix_test_df = pd.DataFrame(confusion_matrix_test).transpose();
with open(results_table_name+method_names[4]+'_ensemble_confusion_matrix.tex', 'w') as texfile:
    texfile.write(confusion_matrix_test_df.to_latex());

# --------------------------------------------------------------------------- #
# Lexus vs Mercedes-Benz classification
comparison = 1;

print('Classifying %s vs. %s' %(names_1[comparison],names_2[comparison]))

model = 0;
y_test_1 = np.load(script_names[0]+'_'+names_1[comparison]+'_'+names_2[comparison]+model_names[model]+'y_test.npy')
y_pred_test_continuous_arr_1 = np.zeros([num_methods,len(y_test_1)],dtype=np.float64);

y_pred_train_continuous_1 = np.load(script_names[0]+'_'+names_1[comparison]+'_'+names_2[comparison]+model_names[model]+'y_pred_train_continuous.npy')
y_pred_train_continuous_arr_1 = np.zeros([num_methods,len(y_pred_train_continuous_1)],dtype=np.float64);

y_train_1 = np.load(script_names[0]+'_'+names_1[comparison]+'_'+names_2[comparison]+model_names[model]+'y_train.npy')

for script in range(0,len(script_names)):    
    npy_file_name = script_names[script]+'_'+names_1[comparison]+'_'+names_2[comparison]+model_names[script];
    y_pred_test_continuous_arr_1[script,:] = np.ravel(np.load(npy_file_name+'y_pred_test_continuous.npy'));
    y_pred_train_continuous_arr_1[script,:] = np.ravel(np.load(npy_file_name+'y_pred_train_continuous.npy'));

r_y_pred_train_1 = np.dot(y_train_1.astype(np.float32), y_pred_train_continuous_arr_1[0,:])/len(y_pred_train_continuous_arr_1[0,:]);
r_y_pred_train_2 = np.dot(y_train_1.astype(np.float32), y_pred_train_continuous_arr_1[1,:])/len(y_pred_train_continuous_arr_1[1,:]);
r_y_pred_train_3 = np.dot(y_train_1.astype(np.float32), y_pred_train_continuous_arr_1[2,:])/len(y_pred_train_continuous_arr_1[2,:]);
r_y_pred_train_4 = np.dot(y_train_1.astype(np.float32), y_pred_train_continuous_arr_1[3,:])/len(y_pred_train_continuous_arr_1[3,:]);

# Correlation vector
r_mat = np.array([[r_y_pred_train_1],[r_y_pred_train_2],[r_y_pred_train_3],[r_y_pred_train_4]])

# normalized correlation vector
a_weights = r_mat/np.sum(r_mat);

# binarize predictions
y_pred_test_1_b = (y_pred_test_continuous_arr_1[0,:] > 0.5).astype(int)
y_pred_test_2_b = (y_pred_test_continuous_arr_1[1,:] > 0.5).astype(int)
y_pred_test_3_b = (y_pred_test_continuous_arr_1[2,:] > 0.5).astype(int)
y_pred_test_4_b = (y_pred_test_continuous_arr_1[3,:] > 0.5).astype(int)

# model average
prob_arr = np.transpose(y_pred_test_continuous_arr_1);
y_pred_test_continuous_mean = np.dot(prob_arr,a_weights);

y_pred_test_mean_b = (y_pred_test_continuous_mean > 0.5).astype(int);

# compute accuracies
y_pred_test_1_b_acc = np.mean(y_pred_test_1_b.ravel() == y_test_1)
y_pred_test_2_b_acc = np.mean(y_pred_test_2_b.ravel() == y_test_1)
y_pred_test_3_b_acc = np.mean(y_pred_test_3_b.ravel() == y_test_1)
y_pred_test_4_b_acc = np.mean(y_pred_test_4_b.ravel() == y_test_1)
y_pred_test_mean_acc = np.mean(y_pred_test_mean_b.ravel() == y_test_1)

columns_names = ['Accuracy'];
method_names = ['Model 1','Model 2','Model 3','Model 4','Ensemble'];

data_results = np.array([y_pred_test_1_b_acc,y_pred_test_2_b_acc,y_pred_test_3_b_acc,y_pred_test_4_b_acc,y_pred_test_mean_acc])
df_table_acc = pd.DataFrame(data = data_results,index=method_names, columns=columns_names)

print('Binary classification task %d' %(comparison+1))
print(df_table_acc)

results_table_name = full_name+'_'+names_1[comparison]+'_'+names_2[comparison];

with open(results_table_name+'_accuracy.tex', 'w') as texfile:
    texfile.write(df_table_acc.to_latex())

report = classification_report(y_test_1,y_pred_test_1_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[0]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())
    
report = classification_report(y_test_1,y_pred_test_2_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[1]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())    
    
report = classification_report(y_test_1,y_pred_test_3_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[2]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())
    
report = classification_report(y_test_1,y_pred_test_4_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[3]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())

report = classification_report(y_test_1,y_pred_test_mean_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[4]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())   

confusion_matrix_test = confusion_matrix(y_test_1,y_pred_test_mean_b);
confusion_matrix_test_df = pd.DataFrame(confusion_matrix_test).transpose();
with open(results_table_name+method_names[4]+'_ensemble_confusion_matrix.tex', 'w') as texfile:
    texfile.write(confusion_matrix_test_df.to_latex());

# --------------------------------------------------------------------------- #
# Honda vs Toyota classification
comparison = 2;

model = 0;
print('Classifying %s vs. %s' %(names_1[comparison],names_2[comparison]))

y_test_1 = np.load(script_names[0]+'_'+names_1[comparison]+'_'+names_2[comparison]+model_names[model]+'y_test.npy')
y_pred_test_continuous_arr_1 = np.zeros([num_methods,len(y_test_1)],dtype=np.float64);

y_pred_train_continuous_1 = np.load(script_names[0]+'_'+names_1[comparison]+'_'+names_2[comparison]+model_names[model]+'y_pred_train_continuous.npy')
y_pred_train_continuous_arr_1 = np.zeros([num_methods,len(y_pred_train_continuous_1)],dtype=np.float64);

y_train_1 = np.load(script_names[0]+'_'+names_1[comparison]+'_'+names_2[comparison]+model_names[model]+'y_train.npy')

for script in range(0,len(script_names)):    
    npy_file_name = script_names[script]+'_'+names_1[comparison]+'_'+names_2[comparison]+model_names[script];
    y_pred_test_continuous_arr_1[script,:] = np.ravel(np.load(npy_file_name+'y_pred_test_continuous.npy'));
    y_pred_train_continuous_arr_1[script,:] = np.ravel(np.load(npy_file_name+'y_pred_train_continuous.npy'));

r_y_pred_train_1 = np.dot(y_train_1.astype(np.float32), y_pred_train_continuous_arr_1[0,:])/len(y_pred_train_continuous_arr_1[0,:]);
r_y_pred_train_2 = np.dot(y_train_1.astype(np.float32), y_pred_train_continuous_arr_1[1,:])/len(y_pred_train_continuous_arr_1[1,:]);
r_y_pred_train_3 = np.dot(y_train_1.astype(np.float32), y_pred_train_continuous_arr_1[2,:])/len(y_pred_train_continuous_arr_1[2,:]);
r_y_pred_train_4 = np.dot(y_train_1.astype(np.float32), y_pred_train_continuous_arr_1[3,:])/len(y_pred_train_continuous_arr_1[3,:]);

# Correlation vector
r_mat = np.array([[r_y_pred_train_1],[r_y_pred_train_2],[r_y_pred_train_3],[r_y_pred_train_4]])

# normalized correlation vector
a_weights = r_mat/np.sum(r_mat);

# binarize predictions
y_pred_test_1_b = (y_pred_test_continuous_arr_1[0,:] > 0.5).astype(int)
y_pred_test_2_b = (y_pred_test_continuous_arr_1[1,:] > 0.5).astype(int)
y_pred_test_3_b = (y_pred_test_continuous_arr_1[2,:] > 0.5).astype(int)
y_pred_test_4_b = (y_pred_test_continuous_arr_1[3,:] > 0.5).astype(int)

# model average
prob_arr = np.transpose(y_pred_test_continuous_arr_1);
y_pred_test_continuous_mean = np.dot(prob_arr,a_weights);

y_pred_test_mean_b = (y_pred_test_continuous_mean > 0.5).astype(int);

# compute accuracies
y_pred_test_1_b_acc = np.mean(y_pred_test_1_b.ravel() == y_test_1)
y_pred_test_2_b_acc = np.mean(y_pred_test_2_b.ravel() == y_test_1)
y_pred_test_3_b_acc = np.mean(y_pred_test_3_b.ravel() == y_test_1)
y_pred_test_4_b_acc = np.mean(y_pred_test_4_b.ravel() == y_test_1)
y_pred_test_mean_acc = np.mean(y_pred_test_mean_b.ravel() == y_test_1)

columns_names = ['Accuracy'];
method_names = ['Model 1','Model 2','Model 3','Model 4','Ensemble'];

data_results = np.array([y_pred_test_1_b_acc,y_pred_test_2_b_acc,y_pred_test_3_b_acc,y_pred_test_4_b_acc,y_pred_test_mean_acc])
df_table_acc = pd.DataFrame(data = data_results,index=method_names, columns=columns_names)

print('Binary classification task %d' %(comparison+1))
print(df_table_acc)

results_table_name = full_name+'_'+names_1[comparison]+'_'+names_2[comparison];

with open(results_table_name+'_accuracy.tex', 'w') as texfile:
    texfile.write(df_table_acc.to_latex())

with open(results_table_name+'_accuracy.tex', 'w') as texfile:
    texfile.write(df_table_acc.to_latex())

report = classification_report(y_test_1,y_pred_test_1_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[0]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())
    
report = classification_report(y_test_1,y_pred_test_2_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[1]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())    
    
report = classification_report(y_test_1,y_pred_test_3_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[2]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())
    
report = classification_report(y_test_1,y_pred_test_4_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[3]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex())

report = classification_report(y_test_1,y_pred_test_mean_b, target_names=[names_1[comparison],names_2[comparison]], output_dict=True);
classification_report_df = pd.DataFrame(report).transpose();
with open(results_table_name+method_names[4]+'.tex', 'w') as texfile:
    texfile.write(classification_report_df.to_latex()) 
    
confusion_matrix_test = confusion_matrix(y_test_1,y_pred_test_mean_b);
confusion_matrix_test_df = pd.DataFrame(confusion_matrix_test).transpose();
with open(results_table_name+method_names[4]+'_ensemble_confusion_matrix.tex', 'w') as texfile:
    texfile.write(confusion_matrix_test_df.to_latex());
# --------------------------------------------------------------------------- #

