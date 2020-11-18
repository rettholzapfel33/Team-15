#   Taylor Chase Hunter #
#   This code modifies the num-data.csv file    #
#   and applies a regression tree to the data   #

import csv
import pandas as pd
import numpy as np
from os import path


basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "Data", "num-data.csv"))
data = pd.read_csv(filepath)

#   Clear away uncessary columns #
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data = data.drop(['Participant_ID'], axis=1)
data = data.drop('Group', axis=1)
data = data.drop('Treatment', axis=1)
data = data.drop('Treatment_Time', axis=1)
data = data.drop('PP_QC', axis=1)
data = data.drop('EDA_QC', axis=1)
data = data.drop('BR_QC', axis=1)
data = data.drop('Chest_HR_QC', axis=1)





print(data)



