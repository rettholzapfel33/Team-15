#omar
# Run PCA on feature set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# create useful lists
fpath = '../Data/stand_num_Xs.csv'
tpath = '../Data/num-data.csv'
col_names = ['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Chest_HR_QC','Wrist_HR_QC','Age','Gender']
features_l = ['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Age','Gender']
targets_l = ['Chest_HR_QC','Wrist_HR_QC']

# load dataset into Pandas DataFrame
X = pd.read_csv(fpath, usecols=features_l)
Y = pd.read_csv(tpath, usecols=targets_l)

# remove rows with missing values
df_total = pd.concat([X, Y], axis=1)
df_total = df_total.dropna(axis=0, how='any')


# separate back into features and targets
s_f = df_total[features_l]
s_t = df_total[targets_l]
X = s_f.copy()
Y = s_t.copy()



# run pca
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4'])

finalDf = pd.concat([principalDf, Y[targets_l]], axis = 1)

#Plot PCA
""" fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [catergory1, category2]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Chest_HR_QC'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show() """
###


print("How much of our variance is explained?")
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
print()
print() 

print("Which features matter most?")
print(abs(pca.components_))