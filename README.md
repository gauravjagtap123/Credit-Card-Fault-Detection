# Credit-Card-Fault-Detection
As the world is moving toward digitization more security loopholes have been developing every day. And online transaction has become a crucial part of our life and so the threat of cybersecurity. So we need a solution which can track the abnormality in our transaction so if a transaction is fraud we can stop it. In this project we have build a model which will predict credit card fraud transaction based on the past dataset. 

Now taking out the data which is the first step of a Data Science project also called as Data collection, so the data which we have used is taken from Kaggle.com which are real transaction made through credit card in the month of september by European cardholders. The data is highly unbalanced as the legit transaction is 275190 whereas the fraud transaction are only 473 out of 284,807 total transactions. The fraud transaction accounts only 0.172% of the total transaction. So our first task was to handle this unbalanced data.

After understanding the data we must load all the required libraries from performing this project. Now comes the interesting part of data investigation, which starts which Data Analysis of Data which include finding the missing and duplicate values and removing it, and understanding various features of the dataset and the relationships between them.  After Data Analysis we handle the unbalanced dataset using under-sampling technique to uniformly distribute the data. Thereafter our data is cleaned and ready for model development.  Before model development we must split our data into training and testing data so we can feed our training data to different Machine learning models and train the models and after that we will evaluate our models with the testing data.  
So how comes the model development for which we have done all this hardwork. We will develope different machine model and find there accuracy score F1 score to evaluate which model has the highest precision.  During our model development we train and build the following model: 

Logistic Regression.
Decision Tree. 
K-Nearest Neighbors 
Random Forest 

But Logistic Regression got the highest accuracy score of 94.93% and F1 score of 94.82%.   We got 94.93% accuracy through Logistic Regression in our Credit Card Fraud Detection Project.  Hope you like it.....!!!
