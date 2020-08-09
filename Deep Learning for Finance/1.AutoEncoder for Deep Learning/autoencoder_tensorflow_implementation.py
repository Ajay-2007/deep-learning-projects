import tensorflow.compat.v1 as tf
from sklearn.utils import shuffle

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

tf.disable_v2_behavior()
data = pd.read_csv('dj30_10y.csv', sep=',', engine='python')
assets = data.columns.values[1:].tolist()

# excluding Date column
data = data.iloc[:, 1:]

# Load index
index = pd.read_csv('dj30_index_10y.csv', sep=',', engine='python')
index = index.iloc[-data.values.shape[0]:, 1:]

# Normalize data
scaler = MinMaxScaler([0.1, 0.9])
data_X = scaler.fit_transform(data)
scaler_index = MinMaxScaler([0.1, 0.9])
index = scaler_index.fit_transform(index)

# Number of components
N_COMPONENTS = 3

## Autoencoder - Keras
# Network hyperparameters
n_inputs = len(assets)
n_core = N_COMPONENTS
n_outputs = n_inputs

# Testing in-sample
X_train = data_X
X_test = data_X

initializer = tf.initializers.glorot_normal()
W1 = tf.Variable(initializer([n_inputs, n_core]))
W2 = tf.transpose(W1)
b1 = tf.Variable(tf.zeros([n_core]))
b2 = tf.Variable(tf.zeros([n_outputs]))


# Building the encoder
def encoder(x):
    return tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1))


# Building the decoder
def decoder(x):
    return tf.nn.sigmoid(tf.add(tf.matmul(x, W2), b2))


X = tf.placeholder("float", [None, n_inputs])
Y = tf.placeholder("float", [None, n_inputs])

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
y_true = X

# Training parameters
lr = 0.01
epochs = 20
batch_size = 1

mse = tf.losses.mean_squared_error(y_true, y_pred)
optimizer = tf.train.AdamOptimizer(lr).minimize(mse)

# Start Training
# Start a new TF session
with tf.Session() as sess:
    # Initialize the network
    sess.run(tf.global_variables_initializer())

    # Training
    for i in range(epochs):
        X_train1 = shuffle(X_train)
        for j in range(X_train.shape[0] // batch_size):
            batch_y = X_train1[j * batch_size: j * batch_size + batch_size, :]
            batch_x = X_train1[j * batch_size: j * batch_size + batch_size, :]
            _, loss_value = sess.run([optimizer, mse], feed_dict={X: batch_x, Y: batch_y})

        # Display loss
        print('Epoch: %i -> Loss: %f' % (i, loss_value))

    # Make predictions
    y_pred_AE_tf = sess.run(decoder_op, feed_dict={X: X_train, Y: X_train})
    print('Test Error: %f' % tf.losses.mean_squared_error(X_train, y_pred_AE_tf).eval())
