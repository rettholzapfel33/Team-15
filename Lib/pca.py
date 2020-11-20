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
features_total = ['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Age','Gender']
features_l = ['Participant_ID','Treatment','Treatment_Time','Task','PP_QC']
targets_total = ['Chest_HR_QC','Wrist_HR_QC']
targets_l = ['Chest_HR_QC']

# load dataset into Pandas DataFrame
X = pd.read_csv(fpath, usecols=features_total)
Y = pd.read_csv(tpath, usecols=targets_total)

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
             , columns = ['pc1','pc2','pc3','pc4'])

finalDf = pd.concat([principalDf, Y[['Chest_HR_QC']]], axis=1)


print("How much of our variance is explained?")
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
print()
print() 


print("Which features matter most?")
print(abs(pca.components_))
print()

#Print sum of features importance to find maximum contribution
df = []
i = 0
for ele in pca.components_[0]:
    df.append(abs(ele) + abs(pca.components_[1][i]))
    i+=1
print("How much does each feature contribute overall")
print(df)