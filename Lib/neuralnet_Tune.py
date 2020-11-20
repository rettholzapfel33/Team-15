# Use scikit-learn to grid search the number of neurons
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from keras.constraints import maxnorm

# (0) Hide as many warnings as possible!
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')
tf.compat.v1.disable_eager_execution()

# Load the dataset
heartRate = pd.read_csv('../Data/num-data.csv')

# Sample 5,000 rows of random data
heartRate = heartRate.sample(n=5000, axis=0)

heartRate = heartRate.dropna(axis=0, how='any')

# Split the data into features and targets
features = ['Treatment', 'Treatment_Time',
    'Task', 'EDA_QC', 'BR_QC', 'Age', 'Gender']
targets = ['Chest_HR_QC']

heartX = heartRate[features]
heartY = heartRate[targets]

heartX = heartX.iloc[1:]
heartY = heartY.iloc[1:]

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

# Build Keras models.
def DynamicModel(neurons=1, activation_func='sigmoid'):
    """ A sequential Keras model that has an input layer, one 
        hidden layer with a dymanic number of units, and an output layer."""
    model = Sequential()
    model.add(Dense(neurons, input_dim=7, activation=activation_func, name='layer_1'))
    model.add(Dense(1, activation='sigmoid', name='output_layer'))
     
    # Don't change this!
    model.compile(loss="mean_squared_error",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model


# Evaluation + HyperParameter Search
model = KerasClassifier(
    build_fn=DynamicModel, 
    epochs=200, 
    batch_size=20, 
    verbose=0)

# Define a set of unit numbers (i.e. "neurons") and activation functions
param_grid = [
    {
        'activation_func': ['linear', 'sigmoid', 'relu', 'tanh'], 
        'neurons': [10, 15, 20, 25, 30]
    }
]

# Send the Keras model through GridSearchCV, and evaluate the performnce of every option in 
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# Print out a summarization of the results.
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))