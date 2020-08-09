import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

df = pd.read_csv("african_crises.csv")
df = shuffle(df, random_state=11)
df.reset_index(inplace=True, drop=True)

# data labeling
df.banking_crisis = df.banking_crisis.replace('crisis', np.nan)
df.baking_crisis = df.banking_crisis.fillna(1)
df.banking_crisis = df.banking_crisis.replace('no_crisis', np.nan)
df.banking_crisis = df.banking_crisis.fillna(0)

# data cleaning and scaling
df.drop(['cc3', 'country'], inplace=True, axis=1)
df_scaled = preprocessing.scale(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled.banking_crisis = df.banking_crisis
df = df_scaled

# defining the input data X, and the desired results, y
X = df.loc[:, df.columns != 'banking_crisis']
y = df.loc[:, 'banking_crisis']

# breaking data into training data, validation data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# constructing a Multilayer Perceptron
model = Sequential()
model.add(Dense(32, input_shape=(11,), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# training the network
model.fit(X_train, y_train, epochs=10, verbose=2)

scores = model.evaluate(X_test, y_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1] * 100))
scores = model.evaluate(X_train, y_train)
print("Testing Accuracy: %.2f%%\n" % (scores[1] * 100))
