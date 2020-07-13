# Udacity Arvato Capstone Project

This is the Arvato Capstone project of Udacity. Data from a online sales company in Germany have been provided. The data are as below

- Customers data
- Germany Population data with the same features as Customers data
- Mailout training data. This data shares the common features with customers but has an additional feature, if the person recieving the email became a customer or not.
- Mailout test data. This is a unlabeled data to be predicted.

## Goals

Before doing any analysis the data had to be cleaned. ETL pipeline workflow was used to accomplish this task.

With the clean data the aim was to perform clustering on the Customers and the Germany population data and compare the results.

Train a model using the Mailout training data and do a prediction of Mailout test data. The results of this prediction is submitted to Kaggle and gotten 0.51 as a result.

## Requirements

The libraries required to run the code in addition to the Anaconda standard installation are listed below.
- numpy
- pandas
- matplotlib
- seaborn
- pickle
- copy
- warnings
- sklearn (scikit-learn)
- progressbar

A custom module is also created called ETL. This module folder needs to be in the same directory as the Jupyter Notebook file.

For the code to function as is the data files should be in the same directory as the Jupyter Notebook.

## Future Work
The mailout train data was very sparse on the perspective of positive returns in the target. While training only equal numbers of possitive and negative target variables are selected to remove bias. This led to a great loss in data. This can be mnitigated by assuming the previous customers dataset as positive returns of a fictional mailout campaign. The model can be retrained with this higher amount of data.

Also other algorithms to train the model can be tested. Here I only used GradientBoostingClassifier from sklearn.

The report is also shared on a Medium article at:
https://medium.com/p/eebe1295222a/edit
