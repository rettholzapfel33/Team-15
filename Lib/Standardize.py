import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from os import path

#   Get file path   #
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "Data", "num-data.csv"))

df = pd.read_csv(filepath)

# Define the parameters we want to use.

features = ['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Age','Gender']

# Separate our features from our targets.
X = df[features]
Y = df[['Chest_HR_QC']]

#Standardize the X's
scaler = StandardScaler().fit(X.values)
standardizedXs = scaler.transform(X.values)

#Create csv
dX = pd.DataFrame(standardizedXs, index = None, columns = ['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Age','Gender'])
dX.to_csv('../Data/stand_num_Xs.csv', index = False)
