# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 02:59:14 2020

Multiclass classification

Ensembling the classifiers by averaging

@author: Ashiv Hans Dhondea
"""

# --------------------------------------------------------------------------- #

# Import the necessary packages
# numpy for linear algebra, cv2 for image processing
#  glob and os to navigate directories
import numpy as np    
import pandas as pd
from sklearn.metrics import confusion_matrix 

# matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['DejaVu Sans']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params);
import seaborn as sn

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

full_name = script_name+'_';
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

for k in range(len(curated_file_list)):
    filename = str(curated_file_list[k]);   
    dataset_y_labels[k] = filename[2:].split("_")[0];
 
curated_brands, curated_brands_freq = np.unique(np.asarray(dataset_y_labels),return_counts=True);
# --------------------------------------------------------------------------- #
script_names = ['main_20_multiclass_00','main_21_multiclass_00','main_22_multiclass_00','main_23_multiclass_00','main_24_multiclass_00'];

num_methods = len(script_names);
y_test_1 = np.load(script_names[0]+'_'+'y_test.npy')

#y_pred_train_continuous_1 = np.load(script_names[0]+'_'+'y_pred_train_continuous.npy')
#y_pred_train_continuous_2 = np.load(script_names[1]+'_'+'y_pred_train_continuous.npy')
#y_pred_train_continuous_3 = np.load(script_names[2]+'_'+'y_pred_train_continuous.npy')
#y_pred_train_continuous_4 = np.load(script_names[3]+'_'+'y_pred_train_continuous.npy')
#y_pred_train_continuous_5 = np.load(script_names[4]+'_'+'y_pred_train_continuous.npy')

y_pred_test_continuous_1 = np.load(script_names[0]+'_'+'y_pred_test_continuous.npy');
y_pred_test_continuous_2 = np.load(script_names[1]+'_'+'y_pred_test_continuous.npy');
y_pred_test_continuous_3 = np.load(script_names[2]+'_'+'y_pred_test_continuous.npy');
y_pred_test_continuous_4 = np.load(script_names[3]+'_'+'y_pred_test_continuous.npy');
y_pred_test_continuous_5 = np.load(script_names[4]+'_'+'y_pred_test_continuous.npy');

# --------------------------------------------------------------------------- #
# model average

y_pred_test_continuous = 0.2*(y_pred_test_continuous_1+y_pred_test_continuous_2+y_pred_test_continuous_3+y_pred_test_continuous_4+y_pred_test_continuous_5);
y_pred_test = np.argmax(y_pred_test_continuous,axis=1);

y_pred_test_1 = np.argmax(y_pred_test_continuous_1,axis=1);
y_pred_test_2 = np.argmax(y_pred_test_continuous_2,axis=1);
y_pred_test_3 = np.argmax(y_pred_test_continuous_3,axis=1);
y_pred_test_4 = np.argmax(y_pred_test_continuous_4,axis=1);
y_pred_test_5 = np.argmax(y_pred_test_continuous_5,axis=1);

y_test_enc = np.argmax(y_test_1,axis=1);

y_test_compare_1 = np.equal(y_test_enc,y_pred_test_1);
y_pred_test_accuracy_1 = np.count_nonzero(y_test_compare_1)/np.shape(y_test_compare_1)[0];

y_test_compare_2 = np.equal(y_test_enc,y_pred_test_2);
y_pred_test_accuracy_2 = np.count_nonzero(y_test_compare_2)/np.shape(y_test_compare_2)[0];

y_test_compare_3 = np.equal(y_test_enc,y_pred_test_3);
y_pred_test_accuracy_3 = np.count_nonzero(y_test_compare_3)/np.shape(y_test_compare_3)[0];

y_test_compare_4 = np.equal(y_test_enc,y_pred_test_4);
y_pred_test_accuracy_4 = np.count_nonzero(y_test_compare_4)/np.shape(y_test_compare_4)[0];

y_test_compare_5 = np.equal(y_test_enc,y_pred_test_5);
y_pred_test_accuracy_5 = np.count_nonzero(y_test_compare_5)/np.shape(y_test_compare_5)[0];

y_test_compare_average = np.equal(y_test_enc,y_pred_test);
y_pred_test_accuracy_average = np.count_nonzero(y_test_compare_average)/np.shape(y_test_compare_average)[0];


columns_names = ['Accuracy'];
method_names = ['Method 1','Method 2','Method 3','Method 4','Method 5','Ensemble'];

data_results = np.array([y_pred_test_accuracy_1,y_pred_test_accuracy_2,y_pred_test_accuracy_3,y_pred_test_accuracy_4,y_pred_test_accuracy_5,y_pred_test_accuracy_average])
df_table_acc = pd.DataFrame(data = data_results,index=method_names, columns=columns_names)

print(df_table_acc)

# Store results in tables in LaTeX format
with open(full_name+'accuracy_results.tex', 'w') as texfile:
    texfile.write(df_table_acc.to_latex())
    
# --------------------------------------------------------------------------- #
confusion_matrix = confusion_matrix(y_test_enc, y_pred_test)

confusion_matrix_df = pd.DataFrame(confusion_matrix);
with open(full_name+'confusion_matrix.tex', 'w') as texfile:
    texfile.write(confusion_matrix_df.to_latex())
    
df_cm = pd.DataFrame(confusion_matrix, curated_brands, curated_brands)
df_cm.index.name = r'\textbf{Actual}'
df_cm.columns.name = r'\textbf{Predicted}'
plt.figure(figsize=(10,7))
#sn.set(font_scale=1.4) # for label size
plt.title(r"\textbf{Confusion matrix on Test Set with Ensemble Averaging}",fontsize=12);
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
plt.savefig(full_name+'confusion_matrix.pdf');
plt.show()
# --------------------------------------------------------------------------- #