import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from os import path

#   Get file path   #
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "Data", "num-data.csv"))

df = pd.read_csv(filepath)

# Define the parameters we want to use.

features = ['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Age','Gender','Chest_HR_QC']

# Separate our features from our targets.
d = df[features]
dn = d.dropna()
X = dn[['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Age','Gender']]
Y = dn[['Chest_HR_QC']]

#Standardize the X's
scaler = StandardScaler().fit(X.values)
standardizedXs = scaler.transform(X.values)

#Create csv
dX = pd.DataFrame(np.concatenate([standardizedXs,Y.values], axis=1), index = None, columns = ['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Age','Gender','Chest_HR_QC'])

dX.to_csv('../Data/stand_num_Xs.csv', index = False)
