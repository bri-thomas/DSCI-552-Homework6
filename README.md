DSCI 552 - Homework 6
Brian Thomas

This repository contains my Homework 6 Jupyter notebook for DSCI 552.

Contents
APS Failure data preparation, median imputation, and exploratory plots.
Random forests with and without class weighting.
L1-regularized XGBoost linear classifiers with and without SMOTE, using stratified 5-fold cross-validation to select alpha.
Confusion matrices, ROC curves, AUC, misclassification rates, and random forest OOB error.
ISLR exercises 6.6.3, 6.6.5, 8.4.5, and 9.7.3.
The XGBoost implementation uses gblinear with a logistic objective; it does not fit logistic regression inside tree nodes. Imputation, scaling, and SMOTE are applied within the training folds during cross-validation.

Files
Homework 6.ipynb: code, results, plots, and written answers.
data/aps_failure_training_set.csv: provided training data.
data/aps_failure_test_set.csv: provided test data.
data/aps_failure_description.txt: dataset description.
How to Run
Download or clone this repository.

Install the required packages in your Python environment:

pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
Place the dataset files in data/ next to the notebook.

Open Homework 6.ipynb in PyCharm or Jupyter and run the cells from top to bottom. Use the repository root as the working directory.

On macOS, XGBoost may also require OpenMP:

brew install libomp
SMOTE and cross-validation may take several minutes to run.

Dataset
APS Failure at Scania Trucks - UCI Machine Learning Repository

The dataset includes 60,000 training observations, 16,000 test observations, and 170 numeric features. The positive class represents APS-related failures. The provided training and test sets are kept separate.
