from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import binary_crossentropy
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class RBF(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBF, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBF, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def comput_output_shape(self, input_shape):
        return (input_shape[0], self.units)


df = pd.read_csv('african_crises.csv')
df = shuffle(df, random_state=11)
df.reset_index(df, inplace=True, drop=True)

# converting into useful numbers
df.banking_crisis = df.banking_crisis.replace('crisis', np.nan)
df.banking_crisis = df.banking_crisis.fillna(1)
df.banking_crisis = df.banking_crisis.replace('no_crisis', np.nan)
df.banking_crisis = df.banking_crisis.fillna(0)

# removing unnecessary data
df.drop(['cc3', 'country'], axis=1, inplace=True)

# scaling the data
df_scaled = preprocessing.scale(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled['banking_crisis'] = df.banking_crisis
df = df_scaled

# defining the input data, X, and the desired results, y
X = df.loc[:, df.columns != 'banking_crisis']
y = df.loc[:, 'banking_crisis']

# breaking data into training data, validation data and test_data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = Sequential()

input = Input(shape=(X_train.shape[1],))
rbf = RBF(10, 0.5)(input)
out = Dense(1, activation='sigmoid')(rbf)
model = Model(inputs=input, outputs=out)
model.compile(optimizer='rmsprop', loss=binary_crossentropy, metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, batch_size=256, epochs=30)

scores = model.evaluate(X_train, y_train)

print("Training Accuracy: %.2f%%\n" % (scores[1] * 100))

scores = model.evaluate(X_test, y_test)

print("Testing Accuracy: %.2f%%\n" % (scores[1] * 100))
