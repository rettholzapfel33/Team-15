from pandas.core.reshape.merge import merge
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential #type: ignore
from keras.layers import Dense #type: ignore
from sklearn.preprocessing import LabelBinarizer
from keras.wrappers.scikit_learn import KerasClassifier #type: ignore
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

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
heartRate = heartRate.sample(n=5000, axis=0)

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

# Build Keras Model 1 based on tuning
def BaselineModel():
    model = Sequential()
    model.add(Dense(7, input_dim=7, activation='relu', name='layer_1'))
    model.add(Dense(30, activation='relu', name='layer_2'))
    model.add(Dense(10, activation='sigmoid', name='output_layer'))
     
    # Don't change this!
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model

# Keras Model 2
def AlternativeModel1():
    model = Sequential()
    model.add(Dense(7, input_dim=7, activation='relu', name='layer_1'))
    model.add(Dense(10, activation='sigmoid', name='layer_2'))
    model.add(Dense(5, activation='sigmoid', name='layer_3'))
    model.add(Dense(10, activation='sigmoid', name='output_layer'))
    
    # Don't change this!
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model


# Evaluate the models

# - - Model 1 - - 
estimator = KerasClassifier(
        build_fn=BaselineModel,
        epochs=200, batch_size=20,
        verbose=0)
kfold = KFold(n_splits=5, shuffle=True)
print("- - - - - - - - - - - - - ")
for i in range(0,10):
    results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    print("(MODEL 1 : RUN " + str(i) +") Performance: mean: %.2f%% std: (%.2f%%)" % (results.mean()*100, results.std()*100))

# - - Model 2 - - 
estimator = KerasClassifier(
        build_fn=AlternativeModel1,
        epochs=200, batch_size=20,
        verbose=0)
kfold = KFold(n_splits=5, shuffle=True)
print("- - - - - - - - - - - - - ")
for i in range(0,10):
    results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    print("(MODEL 2 : RUN " + str(i) +") Performance: mean: %.2f%% std: (%.2f%%)" % (results.mean()*100, results.std()*100))