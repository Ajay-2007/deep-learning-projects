from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input

model = InceptionV3(input_tensor=Input(shape=(224, 224, 3)),
                    weights='imagenet',
                    include_top=True)

print(model.summary())
