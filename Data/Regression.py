import numpy as np
import pandas as pd
import pylab as pl
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn





filename = '../Data/stand_num_Xs.csv'
df = pd.read_csv(filename)


# Separate our features from our targets.

#Total Features = ['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Age','Gender']
features = ['Group','Treatment','Treatment_Time','EDA_QC','BR_QC']

X = df[features]
Y = df[['Chest_HR_QC']]

#Kfold split

kf = KFold(n_splits = 10)
for train_index, test_index in kf.split(range(len(X))):
    x_train, x_test = X.values[train_index], X.values[test_index]
    y_train, y_test = Y.values[train_index], Y.values[test_index]
    y_train_converted = y_train.ravel()
    y_test_converted = y_test.ravel()

    #Train Models
    linReg = LinearRegression(n_jobs = -2).fit(x_train, y_train_converted)
    gradReg = SGDRegressor().fit(x_train, y_train_converted)

    # Now, we run our test data through our trained models.
    # = linReg.predict(x_test)
    #predicted_grad = gradReg.predict(x_test)
    
    linR2 = linReg.score(x_test, y_test_converted)
    gradR2 = gradReg.score(x_test, y_test_converted)
    
    print("Lin R2 Score \t\t-> " + str(linR2))
    print("GradDes R2 Score \t-> " + str(gradR2))
    print()

