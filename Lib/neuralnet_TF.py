import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow.compat.v1 as tf #type: ignore
from tensorflow import keras
tf.disable_v2_behavior() 
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
heartRate = pd.read_csv('../Data/num-data.csv')

# Split the data into features and targets
features = ['Group', 'Treatment', 'Treatment_Time',
    'Task', 'PP_QC', 'EDA_QC', 'BR_QC', 'Age', 'Gender']
targets = ['Chest_HR_QC']

heartX = heartRate[features]
heartY = heartRate[targets]

heartX = heartX.iloc[1:]
heartY = heartY.iloc[1:]

# Using Sklearn to split training and target
x_train, x_test, y_train, y_test = train_test_split(
    heartX, heartY, test_size=0.8, random_state=0)

y_train_converted = y_train.values.ravel()

# Specify the hidden neutrons
num_hidden = 15
num_input = x_train.shape[1]
num_class = y_train.shape[1]

# Specify the weight, bias and keep probability
weights = {
    'h1': tf.Variable(tf.random.normal([num_input, num_hidden])),
    'out': tf.Variable(tf.random.normal([num_hidden, num_class]))
}

biases = {
    'b1': tf.Variable(tf.random.normal([num_hidden])),
    'out': tf.Variable(tf.random.normal([num_class]))
}

keep = tf.placeholder("float")

# Specify the number of epochs
num_epochs = 2000
display_step = 500
batch_size = 32

x = tf.placeholder("float", [None, num_input])
y = tf.placeholder("float", [None, num_class])

# Function to run neural net model
def create_model(x, weights, biases, keep):

    initial = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    initial = tf.nn.sigmoid(initial)
    initial = tf.nn.dropout(initial, keep)
    outer = tf.matmul(initial, weights['out']) + biases['out']

    return outer

# Define cost function and optimizer
predictions = create_model(x, weights, biases, keep)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

# Run session and print results
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(num_epochs):
        avg_cost = 0.0
        total_batch = int(len(x_train) / batch_size)
        x_batches = np.array_split(x_train, total_batch)
        y_batches = np.array_split(y_train, total_batch)
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            _, c = sess.run([optimizer, cost], 
                            feed_dict={
                                x: batch_x, 
                                y: batch_y, 
                                keep: 0.8
                            })
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test, keep: 1.0}))