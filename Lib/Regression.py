import numpy as np
import pandas as pd
import pylab as pl
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.model_selection import train_test_split 
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

#X = df[features]
Y = df[['Chest_HR_QC']]


tstVals = []

#Tests arrays of different values for different testing
#tests = [5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,1e-7]
#tests  = ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
#tests = [0,1,10,100,1000,10000,100000]
tests = [['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC','Age','Gender'],['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','BR_QC','Age'],['Participant_ID','Group','Treatment','Treatment_Time','Task','PP_QC','EDA_QC','BR_QC'], ['Group','Treatment','Treatment_Time','Task','Age','Gender'], ['Participant_ID','Treatment','Treatment_Time','Task','PP_QC'],['Participant_ID', 'PP_QC']]
#tests = [1]

for ele in tests:
    X = df[ele]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=1)
    y_train_converted = y_train.values.ravel()
    y_test_converted = y_test.values.ravel()

    #Train Models
    linReg = LinearRegression(n_jobs = -2).fit(x_train.values, y_train_converted)
    gradReg = SGDRegressor(loss= 'squared_loss', alpha=1e-3, max_iter=2500).fit(x_train.values, y_train_converted)
    ridgReg = Ridge().fit(x_train.values, y_train_converted)
        
    #Each Model below is a variant of the default to be usesd when evaluating different variables
    #gradReg = SGDRegressor(loss= ele, alpha=1e-4, max_iter=2500).fit(x_train.values, y_train_converted)
    #gradReg = SGDRegressor(loss= 'squared_loss', alpha=ele, max_iter=2500).fit(x_train.values, y_train_converted)
    #ridgReg = Ridge(alpha=ele).fit(x_train.values, y_train_converted)
        
    # Now, we run our test data through our trained models.
    #predicted_lin= linReg.predict(x_test)
    #predicted_grad = gradReg.predict(x_test)
        
     
    linR2 = linReg.score(x_test.values, y_test_converted)
    gradR2 = gradReg.score(x_test.values, y_test_converted)
    ridgR2 = ridgReg.score(x_test.values, y_test_converted)
            
    print("Lin R2 Score \t\t-> " + str(linR2))
    print("GradDes R2 Score \t-> " + str(gradR2))
    print("Ridge R2 Score \t\t-> " + str(ridgR2))
    print()
        
    
    #tstVals.append(ridgR2)
    tstVals.append(gradR2)
    

'''
#R2 alpha
pl.plot(tests, tstVals)
pl.title("Ridge R Squared for Different Alphas")
pl.ylabel("Ridge R Squared Score")
pl.xlabel("alpha")
pl.xscale("log")
pl.show()
'''
'''
#R2 loss func
pl.plot(tests, tstVals)
pl.title("Gradient Descent R Squared for Different Loss Functions")
pl.ylabel("Gradient Descent R Squared Score")
pl.xlabel("Loss Function")
pl.show()
'''
'''
#R2 train rate
pl.plot(tests, tstVals)
pl.title("Gradient Descent R Squared for Different Alphas")
pl.ylabel("Gradient Descent R Squared Score")
pl.xscale("log")
pl.xlabel("Training Rate")
pl.show()
'''