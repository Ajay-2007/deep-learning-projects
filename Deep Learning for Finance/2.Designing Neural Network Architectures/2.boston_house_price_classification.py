from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# load dataset
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# first we fit the scaler on the training dataset
scaler = StandardScaler()
scaler.fit(X_train)

# then we call the transform method to scale both the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# create model
model = Sequential()
model.add(Dense(8, input_shape=[X_train.shape[1]], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# compile model
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
print(model.summary())

# model training and scoring
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100)
print(model.evaluate(X_test_scaled, y_test))

# summarize history for accuracy
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model mean absolute error')
plt.ylabel('mean absolute error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
