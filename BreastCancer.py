#machine learning can dramatically increase accuracy of determining malignant/benign tumors
#91% versus 79% regular physician

#teach machine how to classify malignant/benign without human intervention and classify new images moving forward

#support vector machines uses "max margin hyperplane" between the two "support vectors" (those which are first at
#separating the data, these are also the most difficult to differentiate bc feature 1 and feature 2 are closest)



#30 features: radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension...etc
# 569 instances
#212 malignant, 357 benign
#target = is it malignant or benign

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#print(cancer)
"""
print(cancer.keys()) #features of data

#print(cancer ['DESCR'])

print(cancer ['target'])

print(cancer ['target_names'])

print(cancer ['feature_names'])
"""

print(cancer['data'].shape) # number of data available and features (parameters)

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                          columns = np.append(cancer['feature_names'], ['target']))#all 30 features with data

print(df_cancer.head())

#have the data successfully loaded, now we visualize the data.

#plot stuff separating malignant versus benign cases for each feature
#malignant cases more common with wider mean radius
#sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])


#sns.countplot(df_cancer['target'])#shows number of malignant/benign cases

#sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

#view correlation
#plt.figure(figsize = (20, 10))# A LOT of data, increase figure size
#sns.heatmap(df_cancer.corr(), annot = True)# high correlation between mean area and mean smoothness, worst area, mean perimeter...etc.



############################################################
#no need to clean data, was clean
#looked at the data, now we will TRAIN THE MODEL

X = df_cancer.drop(['target'], axis = 1)#need dataframe except for target for X values (parameters)

y = df_cancer['target']# 0's and 1's for indentification of malignant/benign

#split from training to testing data, subset of data used for training, then test set that model has never seen

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
#target has 455 values for X_train, split between x_test 114

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()#
svc_model.fit(X_train, y_train)#this trains the data


#trained the model...now we TEST!
#fit the training dataset...need to consider overfitting and underfitting
#generalized model works better than an overfitted model, as it might consider other variables?
#use confusion matrix to evaluate model: True are columns, predictions are output and rows
#misclassified, prediction could be off compared to true class.
#type 1 error: patient has disease based on prediction but based on truth it not true
########not a big deal, because patient is ok
#type 2 error: patient does not have disease based on prediction but the truth says he does!
#######bigger deal, as life or death situation just came up, want to AVOID

y_predict = svc_model.predict(X_test) #predict on trained model

cm = confusion_matrix(y_test, y_predict)

#sns.heatmap(cm, annot = True)#7 type 1 error...can we improve the model
#data normalization values differ (1500 mean area vs .01 mean radius, get all data between 0 and 1)---feature scaling
#X' = (X-Xmin)/(Xmax - Xmin)

#C parameter (optimization on under/overfitting)
# small c = too lose, cost of misclassification low
# large c = high cost of misclassification, could overfit values

#gamma parameter: influence of single training set
# large gamme - close reach, high weight = overfit
#small gamma - far reach, less weight more generalized

min_train = X_train.min()
#normalize model (scale)
range_train = (X_train-min_train).max()
X_train_scaled = (X_train - min_train)/range_train

min_test = X_test.min()
range_test = (X_test-min_test).max()
X_test_scaled = (X_test - min_test)/range_test

#sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)

svc_model.fit(X_train_scaled, y_train)#train model using new scaled data

y_predict = svc_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_predict)

#sns.heatmap(cm, annot = True)
#reduced misclassified to 4 from 7 in prior training data

#print(classification_report(y_test, y_predict))

#part 2 improving model

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)

grid.fit(X_train_scaled, y_train)

grid.best_params_

grid_predictions = grid.predict(X_test_scaled)

#cm = confusion_matrix(y_test, grid_predictions)

#sns.heatmap(cm, annot = True)

print(classification_report(y_test, grid_predictions))










