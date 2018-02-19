# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:18:17 2017

@author: Rishav
"""
from sklearn import linear_model 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from fourier_util import *
MODE='NEW'

X,y=CREATE_DATASET()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#model_linear
if MODE=='NEW':
    model_linear=linear_model.LogisticRegression()
    model_linear.fit(X_train,y_train)
    predict_y=model_linear.predict(X_test)
    cm=confusion_matrix(y_test,predict_y)
    print("\n CONFUSION MATRIX FOR LINEAR \n '{confusion_matrix}'".format(confusion_matrix=cm))
    joblib.dump(model_linear,'LINEAR_MODEL.pkl')

elif MODE=='TRAINING':
    model_linear=joblib.load('LINEAR_MODEL.pkl')
    model_linear.fit(X_train,y_train)
    predict_y=model_linear.predict(X_test)
    cm=confusion_matrix(y_test,predict_y)
    print("\n CONFUSION MATRIX FOR LINEAR \n '{confusion_matrix}'".format(confusion_matrix=cm))
    joblib.dump(model_linear,'LINEAR_MODEL.pkl')


left_data = pd.read_csv('C://Users//Rishav//Desktop//auto_left_test.csv')
sample = left_data[1200:1400]
x = PREPROCESS(sample)
print(model_linear.predict(x))


right_data = pd.read_csv('C://Users//Rishav//Desktop//auto_right_test.csv')
sample = right_data[200:400]
x = PREPROCESS(sample)
print(model_linear.predict(x))
