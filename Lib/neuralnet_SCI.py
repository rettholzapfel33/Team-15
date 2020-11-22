from pandas.core.reshape.merge import merge
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential #type: ignore
from keras.layers import Dense #type: ignore
from sklearn.preprocessing import LabelBinarizer
from keras.wrappers.scikit_learn import KerasRegressor #type: ignore
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# (0) Hide as many warnings as possible!
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')
tf.compat.v1.disable_eager_execution()

# Load the dataset
heartRate = pd.read_csv('../Data/num-data.csv')

# Drop any NAN values
heartRate = heartRate.dropna(axis=0, how='any')

# Use data sampling to reduce the size of the data
heartRate = heartRate.sample(n=500, axis=0)

# Split the data into features and targets
features = ['Treatment', 'Treatment_Time',
    'Task', 'EDA_QC', 'BR_QC', 'Age', 'Gender']
targets = ['Chest_HR_QC']

heartX = heartRate[features]
heartY = heartRate[targets]

# Use pandas to turn heart rate into catagorical data
heartY = pd.cut(heartY['Chest_HR_QC'], bins=[35,50,60,70,80,90,100,110,120,130,140],
                   labels=[1,2,3,4,5,6,7,8,9,10])

# Split the data into train and test data
X_train,X_test,y_train,y_test = train_test_split(heartX,
                                                 heartY,
                                                 test_size=0.30)

# Perform standardization on our data.
scaler = MinMaxScaler(feature_range=(0,1))
X_train = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=X_train.columns,
                               index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test),
                           columns=X_test.columns,
                           index=X_test.index)


#-----------------------------------------------------------------------------
# Define custom loss functions for regression in Keras 
#-----------------------------------------------------------------------------

# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))

#-----------------------------------------------------------------------------
# A Baseline Model
#-----------------------------------------------------------------------------
# built Keras sequential model 
model = Sequential()
# add batch normalization
model.add(BatchNormalization())
# add layer to the MLP for data (404,13) 
model.add(Dense(units=7, activation='relu', input_dim=7))
# add output layer
model.add(Dense(units=1, activation='relu'))
# compile regression model loss should be mean_squared_error //
model.compile(optimizer="Nadam", loss="mean_squared_error", metrics=["mean_squared_error", rmse, r_square])
# enable early stopping based on mean_squared_error
earlystopping=EarlyStopping(monitor="mean_squared_error", patience=40, verbose=1, mode='auto')
# fit model
result = model.fit(np.array(X_train), np.array(y_train), epochs=240, batch_size=5, validation_data=(np.array(X_test), np.array(y_test)))
# get predictions
y_pred = model.predict(X_test)

#-----------------------------------------------------------------------------
# Plot learning curves including R^2 and RMSE
#-----------------------------------------------------------------------------

# plot training curve for R^2 (beware of scale, starts very low negative)
plt.plot(result.history['val_r_square'])
plt.plot(result.history['r_square'])
plt.title('model R^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
           
# plot training curve for rmse
plt.plot(result.history['rmse'])
plt.plot(result.history['val_rmse'])
plt.title('rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import sklearn.metrics, math
print("\n")
print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test,y_pred))
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test,y_pred))
print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_pred)))
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test,y_pred))


#-----------------------------------------------------------------------------
# An Alternative Model
#-----------------------------------------------------------------------------
# built Keras sequential model 
model = Sequential()
# add batch normalization
model.add(BatchNormalization())
# add layer to the MLP for data (404,13) 
model.add(Dense(units=7, activation='relu', input_dim=7))
# a secpmd ;auer
model.add(Dense(units=25, activation='relu', input_dim=7))
# add output layer
model.add(Dense(units=1, activation='relu'))
# compile regression model loss should be mean_squared_error //
model.compile(optimizer="Nadam", loss="mean_squared_error", metrics=["mean_squared_error", rmse, r_square])
# enable early stopping based on mean_squared_error
earlystopping=EarlyStopping(monitor="mean_squared_error", patience=40, verbose=1, mode='auto')
# fit model
result = model.fit(np.array(X_train), np.array(y_train), epochs=240, batch_size=5, validation_data=(np.array(X_test), np.array(y_test)))
# get predictions
y_pred = model.predict(X_test)

#-----------------------------------------------------------------------------
# Plot learning curves including R^2 and RMSE
#-----------------------------------------------------------------------------

# plot training curve for R^2 (beware of scale, starts very low negative)
plt.plot(result.history['val_r_square'])
plt.plot(result.history['r_square'])
plt.title('model R^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
           
# plot training curve for rmse
plt.plot(result.history['rmse'])
plt.plot(result.history['val_rmse'])
plt.title('rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import sklearn.metrics, math
print("\n")
print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test,y_pred))
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test,y_pred))
print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_pred)))
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test,y_pred))