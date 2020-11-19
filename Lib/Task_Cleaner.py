#This is to fix issue 1 #

import csv
import pandas as pd
import numpy as np
from os import path

#   Get file path   #
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "Data", "Merged.csv"))

#   Differnt CSV types  #
file1 = open(filepath)
file2 = open("../Data/New_Merged.csv", 'w')
reader = csv.reader(file1)
writer = csv.writer(file2)

#   Modify Csvs #
for row in reader:
    r5 = row[5]
    r6 = row[6]
    r7 = row[7]

    if row[5] == "":
        r5 = 'None'
    """ if row[6] == "":
        r6 = -1
    if row[7] == "":
        r7 = -1 """

    writer.writerow([row[0], row[1], row[2], row[3], row[4], r5, r6, r7, \
                    row[8], row[9], row[10], row[11], row[12]])

