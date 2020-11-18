#omar
# Convert all data to numerical values
import pandas as pd
import numpy as np

# Read csv file to pd dataframe
print("Reading data")
data = pd.read_csv("../Data/New_Merged.csv")


#for later
num     = data['Unnamed: 0'].values
tt      = data['Treatment_Time'].values
pp      = data['PP_QC'].values
eda     = data['EDA_QC'].values
br      = data['BR_QC'].values
chest   = data['Chest_HR_QC'].values
wrist   = data['Wrist_HR_QC'].values
age     = data['Age'].values
gender  = data['Gender'].values


#to change
tasks   = data['Task'].values
part_id = data['Participant_ID'].values
groups  = data['Group'].values
treats  = data['Treatment'].values

# Convert tasks to discrete nums
n_task = []
n_ID = []
n_group = []
n_treatment = []

t_seen = []
i_seen = []
g_seen = []
tr_seen = []

for i in tasks:
    #task conv
    if i not in t_seen:
        t_seen.append(i)
    n_task.append(t_seen.index(i))

for i in part_id:
    #id conv
    if i not in i_seen:
        i_seen.append(i)
    n_ID.append(i_seen.index(i))

for i in groups:
    #group conv
    if i not in g_seen:
        g_seen.append(i)
    n_group.append(g_seen.index(i))

for i in treats:
    #treatment conv
    if i not in tr_seen:
        tr_seen.append(i)
    n_treatment.append(tr_seen.index(i))


print("Creating new dataframe...")
col_labels = ['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Chest_HR_QC','Wrist_HR_QC','Age','Gender']
num_list = np.array([n_ID, n_group, n_treatment, tt, n_task, pp, eda, br, chest, wrist, age, gender]).T
df = pd.DataFrame(data=num_list, columns=col_labels)

# Write to new file
print("Writing to num-data.csv")
df.to_csv("../Data/num-data.csv")