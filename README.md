# Image analysis toolbox
Toolbox for image analysis pipeline (image pre-processing, feature extraction, feature preprocessing, dimensionality reduction, classification, clustering and regression). It is powered by sklearn and other python libraries.

## Modules
There are currently 5 different modules:
* Feature extraction (haralick features)
* Feature pre-processing (standardization and normalization)
* Dimensionality reduction (PCA)
* Classification (logistic regression, random forests, support vector machines)
* Regression (linear regression)  
More modules (eg. clustering, image pre-processing, etc.) and algorithms in each module will be added later.  

## Structure
All the modules exist in the directory *utils*

## Example usage
An example for the modules is provided in *test.py* where the modules are used as APIs.
It consists of usage of all the existing modules. The dataset used is 
*mats_dataset1* which is a material science dataset consisting of images of dendrites and non-dendritic microstructures.  
Use the code in *test.py* for your purpose. All you need is the *utils* for performing your image analysis on your chosen dataset.  
Please make sure to install the necessary python modules (keras, mahotas, etc.). 
This code is written using python 2.7 distribution of Anaconda.



 
