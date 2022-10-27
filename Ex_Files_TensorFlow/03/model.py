import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training data set from CSV file
training_data_df = pd.read_csv("sales_data_training.csv", dtype=float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_training = training_data_df.drop('total_earnings', axis=1).values
Y_training = training_data_df[['total_earnings']].values

# Load testing data set from CSV file
test_data_df = pd.read_csv("sales_data_test.csv", dtype=float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_testing = test_data_df.drop('total_earnings', axis=1).values
Y_testing = test_data_df[['total_earnings']].values

# All data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well. Create scalers for the inputs and outputs.
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

# It's very important that the training and test data are scaled with the same scaler.
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

# Define model parameters
learning_rate = 0.001
training_epochs = 100
display_step = 5

# Define how many inputs and outputs are in our neural network
number_of_inputs =  9
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# Section One: Define the layers of the neural network itself
tf.compat.v1.disable_eager_execution()

# Input Layer
with tf.variable_scope('input'):
    X = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, number_of_inputs))

# Layer 1
with tf.variable_scope('layer_1'):
    weights =tf.compat.v1.get_variable(name='weights_1', shape=[number_of_inputs, layer_1_nodes], initializer=tf.compat.v1.contrib.layers.xavier_initializer)
    biases = tf.compat.v1.get_variable(name='biases_1', shape=[layer_1_nodes], initializer=tf.compat.v1.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Layer 2
with tf.variable_scope('layer_2'):
    weights =tf.compat.v1.get_variable(name='weights_2', shape=[layer_1_nodes, layer_2_nodes], initializer=tf.compat.v1.contrib.layers.xavier_initializer)
    biases = tf.compat.v1.get_variable(name='biases_2', shape=[layer_2_nodes], initializer=tf.compat.v1.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Layer 3
with tf.variable_scope('layer_3'):
    weights =tf.compat.v1.get_variable(name='weights_3', shape=[layer_2_nodes, layer_3_nodes], initializer=tf.compat.v1.contrib.layers.xavier_initializer)
    biases = tf.compat.v1.get_variable(name='biases_3', shape=[layer_3_nodes], initializer=tf.compat.v1.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)
# Output Layer
with tf.variable_scope('output'):
    weights = tf.compat.v1.get_variable(name='weights_4', shape=[layer_3_nodes, number_of_outputs], initializer=tf.compat.v1.contrib.layers.xavier_initializer)
    biases = tf.compat.v1.get_variable(name='biases_4', shape=[number_of_outputs], initializer=tf.compat.v1.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

# Section Two: Define the cost function of the neural network that will measure prediction accuracy during training

with tf.compat.v1.variable_scope('cost'):
    Y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
    cost = tf.compat.v1.reduce_mean(tf.compat.v1.squared_difference(prediction, Y))


# Section Three: Define the optimizer function that will be run to optimize the neural network

with tf.compat.v1.variable_scope('train'):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)