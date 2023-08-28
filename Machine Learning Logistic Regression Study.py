#!/usr/bin/env python
# coding: utf-8

# Logistic Regression
# 
# Using customer data
# 
# [Data set](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv)

# In[2]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,jaccard_score, log_loss
from mlxtend.plotting import plot_confusion_matrix


# Place dataset into a pandas DataFrame

# In[3]:


path = "ChurnData.csv"
df = pd.read_csv(path)
df.head()


# Select a set of attributes to analyse. Change the churn value from float to int to create a binary target rather than continuous.

# In[4]:


df = df[['tenure','age','address','income','ed','employ','equip','callcard','wireless','churn']]
df['churn'] = df['churn'].astype('int')
df.head()


# Create the features and target data sets, place into numpy arrays. Features are the attributes fed to the machine learning model to train on, comparing the outcomes to the target values

# In[5]:


columns = df.columns[:-1]
features = np.asarray(df[columns])
features = preprocessing.StandardScaler().fit(features).transform(features)
features[0:1]


# In[6]:


target = np.asarray(df['churn'])
print(target.shape)


# Use the SKLearn Module to create a model testing data set and a model training data set from randomly selected rows. In this case the parameter 'test_size' has been set to 0.25, so 25% of the data set will be used for testing, and 75% for training the model.

# In[7]:


x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.25, random_state=4)
print("Data for training: ",x_train.shape, y_train.shape)
print("\nData for testing: ", x_test.shape, y_test.shape)


# Train the model.

# In[8]:


logisticReg = LogisticRegression(C= 0.01, solver='liblinear').fit(x_train,y_train)
logisticReg


# Test the model. This outputs the models predictions of churn values, given the (x) testing data set.

# In[9]:


predictions = logisticReg.predict(x_test)
predictions[0:5]


# This gives the probability of belonging to either the churn 1 or churn 0 class, for each of the models predictions.

# In[10]:


prediction_probability = logisticReg.predict_proba(x_test)
prediction_probability[0:3]


# Compute the Jaccard score to determine the model accuracy. Scores closer to 0 are more accurate.

# In[11]:


jaccard_score(y_test, predictions,pos_label=0)


# Compute the confusion matrix, gives a visual representation of: True positives; True negatives; False positives; and True negatives

# In[16]:


conf_matrix = confusion_matrix(y_test, predictions)
conf_matrix


# In[17]:


# Create a heatmap 
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues', aspect='auto')

# Set x and y ticks
class_labels = ["Churn 1", "Churn 0"]
plt.xticks(np.arange(len(class_labels)), class_labels)
plt.yticks(np.arange(len(class_labels)), class_labels)

# Set labels and title
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')

# Disable grid lines
plt.grid(False)

# Annotate heatmap cells with centered count labels
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j+0, i+0, str(conf_matrix[i, j]), ha='center', va='center', color='black')
plt.show()


# Compute the classification report of the model, including the precision, recall, f-1 score, and support values.
# 
# Precision is a measure of the accuracy provided that a class label has been predicted. It is defined by: precision = TP / (TP + FP)
# 
# Recall is the true positive rate. It is defined as: Recall =  TP / (TP + FN)
# 
# The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. It is a good way to show that a classifer has a good value for both recall and precision.
# 
# The report also shows the average accuracy for this classifier as the average of the F1-score for both classes (0 and 1).

# In[18]:


print(classification_report(y_test, predictions))


# Compute the LogLoss value of the model. It's an indication of how close the model's prediction is the true value. A lower LogLoss represents a smaller deviation of the prediction from the true value.

# In[178]:


print("LogLoss: %0.3f" % log_loss(y_test, prediction_probability))


# Repeat the training and predicting process using a different solver 'sag' rather than 'libinear'

# In[168]:


LR2 = LogisticRegression(C=0.01, solver='sag').fit(x_train,y_train)
yhat_prob2 = LR2.predict_proba(x_test)
print ("LogLoss: : %.3f" % log_loss(y_test, yhat_prob2))


# We can see in this case the Sag solver returns a more accurate model, shown by the lower LogLoss score.
