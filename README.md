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

## File Descriptions

- Arvato_Project_Workbook_supervised.ipynb: Supervised learning study on Mailout data target prediction
- Arvato_Project_Workbook_unsupervised.ipynb: Data cleaning, unsupervised learning and demographoc comparison study
- etl: ETL module used in cleaning the data
- pictures: Plots used in the medium article

## Summary and Future Work

#### Summary
The data provided for population and customers are cleaned thoroughly. During the cleaning process the features in the data are filtered out according to their amount of information (missing data) and our knowledge on the features. The data is then scaled and PCA is applied to reduce the amount of features in the data. Clustering model is trained using the population data and used to predict the customers data. These both cluster outputs are then used to do a demographic comparison between the two data. The clusters representing the population and customers are compared in a way that the most divergent features between the two are investigated.

Further on a supervised learning model is created using the Mailout train data. Several models are compared to one another and KNN is chosen to be the most suitable. Gridseach is used to optimize the hyperparameters of the KNN model in order to improve the predictions. While evaluating the model ROC AUC score is which is selected because it is reliant on the final label outputs of the model. However it was found that the improvements made by grid search in this case did not improve the model any further. The model is then used to predict another dataset called Mailout test and the results are submitted to a Kaggle competition. The resulting score from Kaggle was 0.49118.

#### Future Work
In future work, during the supervised learning the evaluation scores of the tuned model did not give results as expected. This raises the question on how other model application would have performed after propper tuning. During the model selection only bascis were used however to compare the model, all the hyperparameters can be tuned that the full potential of the other models are seen.

Also in order to eliminate the bias in the training data, the data was shaved in a way that the dominant label was reduced to the size of the other. However this removal of bias in the training data does come with a significant cost of loosing data with label 0. In order to eliminate this problem, including the customers data in the training data can be used to increase the size of the training data significantly.
My work can be found in the github repo below,


The report is also shared on a Medium article at:
https://medium.com/@yesilkayacan/customer-clustering-analysis-and-prediction-study-eebe1295222a
