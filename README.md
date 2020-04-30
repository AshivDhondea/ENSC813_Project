# ENSC813_Project
ENSC 813 Deep Learning Sytems in Engineering - Term Project

## Classifying car images in the TCC Dataset
This project implements ConvNets for binary and multi-class classification of car images in the *The Car Connection* dataset.

## Credits
The *The Car Connection* dataset was scraped by [Mr. Nicolas Gervais](https://github.com/nicolas-gervais). The provision of this dataset for free is gratefully acknowledged. The source repo for this dataset is [Predicting car price from scraped data](https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data).

* **Hans Dhondea** - *This repo* - [AshivDhondea](https://github.com/AshivDhondea)

## Report
The project report and its source files can be found in this [repo](https://github.com/AshivDhondea/ENSC813_report).

## User manual
A brief user manual for this code can be found in the [user manual directory](https://github.com/AshivDhondea/ENSC813_report/tree/master/user_manual) of the report [repo](https://github.com/AshivDhondea/ENSC813_report).

## Learned models
Due to GitHub's file size restrictions, most models cannot be committed to this repo. The model and weights files created by [main_22_multiclass_00.py](https://github.com/AshivDhondea/ENSC813_Project/blob/master/scripts/main_22_multiclass_00.py) can be found in the [output_files directory](https://github.com/AshivDhondea/ENSC813_Project/tree/master/output_files).

### Prerequisites
This project was created using a custom *conda* environment called *tfgpu*. 

To re-create this environment, the following command may be run in the *Anaconda Powershell Prompt*: <code>conda create -n tfgpu --ensc813-tfgpu-package-list.txt </code>. More info in the [requirements directory](https://github.com/AshivDhondea/ENSC813_Project/tree/master/requirements).

Python packages used are listed in the file [ensc813-tfgpu-package-list.txt](https://github.com/AshivDhondea/ENSC813_Project/blob/master/requirements/ensc813-tfgpu-package-list.txt) which can be obtained by running the command <code>conda list --export > ensc813-tfgpu-package-list.txt </code>.

### Binary classification
For the binary classification task, we made use of the following ConvNet architectures
![ConvNet architecture](https://github.com/AshivDhondea/ENSC813_Project/blob/master/output_files/table4.png)

### Multi-class classification
We modified the binary classification ConvNets to accommodate multi-class classification problems. We ensembled our ConvNets to obtain an improved multi-class classifier. The following confusion matrix summarizes its classification performance.
![Confusion Matrix](https://github.com/AshivDhondea/ENSC813_Project/blob/master/output_files/confusionmatrix.png)

### Citation
If you use this work, cite it as 
@misc{Hans2020DL,
  author = {Dhondea, A.R.},
  title = {Classifying car images in the TCC Dataset},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AshivDhondea/ENSC813_Project}}
}

### License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/AshivDhondea/ENSC813_Project/blob/master/LICENSE) file for details
