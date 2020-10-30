# This program will merge the data from the Questionnaire data set to the Physiological data set
import pandas as pd 
import numpy as np
import os

# Features you want to merge with Physiological Data
# Keep Participant ID to merge successfully
desired_features = ['Participant_ID', 'Age', 'Gender']

df1 = pd.read_csv('Physiological Data.csv')
df2 = pd.read_csv('Questionnaire Data.csv', usecols=desired_features)

# Remove time feature
df1 = df1.drop('Time', 1)

# Reduce frequecy of treatment_time feature
for idx, row in df1.iterrows():
    if row['Treatment_Time']%5 != 0:
        df1 = df1.drop(idx, 0)
#print(df1)

# Merge Gender and Age to DF1
df3 = pd.merge(df1, df2)
#print(df3)

# Read to csv file
df3.to_csv('Merged.csv')