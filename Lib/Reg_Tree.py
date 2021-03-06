#   Taylor Chase Hunter #
#   This code modifies the num-data.csv file    #
#   and applies a regression tree to the data   #

import csv
import math
import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from os import path
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler

#   File management #
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "Data", "num-data.csv"))
data = pd.read_csv(filepath)

#   Loop through and if no chest_HR then match it to Wrist HR   #
data = data.dropna(axis=0, how='any')
scaler = MinMaxScaler(feature_range=(-1, 1))

#   Clear away uncessary columns #
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data = data.drop(['Participant_ID'], axis=1)
data = data.drop('PP_QC', axis=1)
data = data.drop('EDA_QC', axis=1)
data = data.drop('BR_QC', axis=1)
data = data.drop('Wrist_HR_QC', axis=1)

#   This was used to remove various features for testing    #
#data = data.drop('Group', axis=1)
#data = data.drop('Task', axis=1)
#data = data.drop('Treatment', axis=1)
#data = data.drop('Treatment_Time', axis=1)
#data = data.drop('Age', axis=1)
#data = data.drop('Gender', axis=1)

#   Data seperation #
X = data.drop('Chest_HR_QC', axis=1)
X = scaler.fit_transform(X)
Y = data['Chest_HR_QC']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#   My ML model #
regr_1 = DecisionTreeRegressor(max_depth=9)
regr_1.fit(x_train, y_train)
pred = regr_1.predict(x_test)
score = regr_1.score(x_test, y_test)
print(score)


'''
count = np.arange(start=1, stop=20)
pl.plot(count, score_list)
pl.title("R2 values with tree depth")
pl.ylabel("R2 score")
pl.xlabel("Tree depth")
pl.show()
'''


