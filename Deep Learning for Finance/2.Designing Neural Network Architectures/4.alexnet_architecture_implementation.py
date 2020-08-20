from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

model = Sequential()

# 1st Convolutional Layer

model.add(Conv2D(filters=96, activation='relu', input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4)))

# Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Batch Normalization before passing it to the next layers
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, activation='relu', kernel_size=(11, 11), strides=(1, 1)))

# Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Batch Normalization
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, activation='relu', kernel_size=(3, 3), strides=(1, 1)))

# Batch Normalization
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, activation='relu', kernel_size=(3, 3), strides=(1, 1)))

# Batch Normalization
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, activation='relu', kernel_size=(3, 3), strides=(1, 1)))

# Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Batch Normalization
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())

# 1st Dense Layer
model.add(Dense(4096, activation='relu', input_shape=(224 * 224 * 3,)))

# Add Dropout to prevent over fitting
model.add(Dropout(0.4))

# Batch Normalization
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096, activation='relu'))

# Add Dropout
model.add(Dropout(0.4))

# 3rd Dense Layer
model.add(Dense(1000, activation='relu'))

# Add Dropout
model.add(Dropout(0.4))

# Batch Normalization
model.add(BatchNormalization())

# Output Layer
model.add(Dense(17, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
