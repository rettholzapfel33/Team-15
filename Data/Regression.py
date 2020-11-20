import numpy as np
import pandas as pd
import pylab as pl
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from statistics import mean as mn


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



filename = '../Data/stand_num_Xs.csv'
#filename = '../Data/num-data.csv'
df = pd.read_csv(filename)

#Total Features = ['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Age','Gender']
features = ['Participant_ID','Group','Treatment','Treatment_Time','EDA_QC','BR_QC']
total_cols = ['Participant_ID','Group','Treatment','Treatment_Time','EDA_QC','BR_QC','Chest_HR_QC']




#Drop Na values if using num-data.csv
#df = df.dropna(axis=0, how='any')

# Separate our features from our targets.

#Total Features = ['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Age','Gender']
features = ['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Age','Gender']

X = df[features]
Y = df[['Chest_HR_QC']]

avgg = []
tstVals = []
alphas = [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
kf = KFold(n_splits = 10)
for alph in alphas:
    max = -30
    i = 1
    lin_pscores = []
    grad_pscores = []
    for train_index, test_index in kf.split(range(len(X))):
        x_train, x_test = X.values[train_index], X.values[test_index]
        y_train, y_test = Y.values[train_index], Y.values[test_index]
        y_train_converted = y_train.ravel()
        y_test_converted = y_test.ravel()

        #Train Models
        linReg = LinearRegression(n_jobs = -2).fit(x_train, y_train_converted)
        gradReg = SGDRegressor(loss= 'squared_loss', alpha=alph, max_iter=2500).fit(x_train, y_train_converted)
        ridgReg = Ridge().fit(x_train, y_train_converted)
        
        # Now, we run our test data through our trained models.
        #predicted_lin= linReg.predict(x_test)
        #predicted_grad = gradReg.predict(x_test)
        
     
        linR2 = linReg.score(x_test, y_test_converted)
        gradR2 = gradReg.score(x_test, y_test_converted)
        ridgR2 = ridgReg.score(x_test, y_test_converted)
        
        if gradR2 > max:
            max = gradR2
            
        print("Fold: " + str(i))
        i+=1
        print("Lin R2 Score \t\t-> " + str(linR2))
        print("GradDes R2 Score \t-> " + str(gradR2))
        print("Ridge R2 Score \t\t-> " + str(ridgR2))
        print()
        
        lin_pscores.append(linR2)
        grad_pscores.append(gradR2)
    
    tstVals.append(max)
    avgg.append(mn(grad_pscores))
    
#Avg R2
pl.plot(alphas, avgg)
pl.title("Avg Gradient Descent R Squared for Different Alphas")
pl.ylabel("Avg Gradient Descent R Squared Score")
pl.xlabel("Training Rate")
pl.show()

#Max R2
pl.plot(alphas, tstVals)
pl.title("Max Gradient Descent R Squared for Different Alphas")
pl.ylabel("Gradient Descent R Squared Score")
pl.xlabel("Training Rate")
pl.show()
