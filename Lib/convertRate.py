import pandas as pd 
import numpy as np
import os

heart = pd.read_csv('../Data/num-data.csv')

# Use pandas to turn heart rate into catagorical data
heart = pd.cut(heart.Chest_HR_QC, bins=[35,50,60,70,80,90,100,110,120,130,140],
                   labels=[1,2,3,4,5,6,7,8,9,10])

# Drop any NAN values
heart = heart.dropna(axis=0, how='any')

heart.to_csv('../Data/Categorical.csv')