from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# Instantiate an emtpy model
model = Sequential()

# C1 Convolutional Layers
model.add(Conv2D(filters=6,
                 kernel_size=5,
                 strides=1,
                 activation='relu',
                 input_shape=(32, 32, 1),
                 padding='valid'))

# S1 Pooling Layer
model.add(MaxPooling2D(pool_size=2, strides=2))

# C2 Convolutional Layers
model.add(Conv2D(filters=6,
                 kernel_size=5,
                 strides=1,
                 activation='relu',
                 input_shape=(14, 14, 6),
                 padding='valid'))

# S2 Pooling Layer
model.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the output to feed the fully connected layers
model.add(Flatten())

# FC1 Fully Connected Layer
model.add(Dense(units=120, activation='relu'))

# FC2 Fully Connected Layer
model.add(Dense(units=84, activation='relu'))

# FC3 Fully Connected Layer
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
