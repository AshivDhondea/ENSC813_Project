# ENSC813_Project
ENSC 813 Deep Learning Sytems in Engineering - Term Project

## Classifying car images in the TCC Dataset
This project implements ConvNets for binary and multi-class classification of car images in the *The Car Connection* dataset.

## Credits
The *The Car Connection* dataset was scraped by [Mr. Nicolas Gervais](https://github.com/nicolas-gervais). The provision of this dataset for free is gratefully acknowledged. The source repo for this dataset is [Predicting car price from scraped data](https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data).

* **Hans Dhondea** - *This repo* - [AshivDhondea](https://github.com/AshivDhondea)

### Prerequisites
This project was created using a custom *conda* environment called *tfgpu*. 

To re-create this environment, the following command may be run in the *Anaconda Powershell Prompt*: <code>conda create -n tfgpu --ensc813-tfgpu-package-list.txt </code>

Python packages used are listed in the file ensc813-tfgpu-package-list.txt which can be obtained by running the command <code>conda list --export > ensc813-tfgpu-package-list.txt </code>.

### License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/AshivDhondea/ENSC813_Project/blob/master/LICENSE) file for details
