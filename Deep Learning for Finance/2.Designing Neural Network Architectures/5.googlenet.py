from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate
from tensorflow.python.keras.regularizers import l2

inception_1x1_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                       activation='relu', kernel_regularizer=l2(0.0002))(input)

inception_3x3_reduce = Conv2D(filters=96, kernel_size=(1, 1), padding='same',
                              activation='relu', kernel_regularizer=l2(0.0002))(input)

inception_3x3_conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                            activation='relu', kernel_regularizer=l2(0.0002))(inception_3x3_reduce)

inception_5x5_reduce = Conv2D(filters=16, kernel_size=(1, 1), padding='same',
                              activation='relu', kernel_regularizer=l2(0.0002))(input)

inception_5x5_conv = Conv2D(filters=32, kernel_size=(5, 5), padding='same',
                            activation='relu', kernel_regularizer=l2(0.0002))(inception_5x5_reduce)

inception_3x3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)

inception_3x3_pool_proj = Conv2D(filters=32, kernel_size=(1, 1), padding='same',
                                 activation='relu', kernel_regularizer=l2(0.0002))(inception_3x3_pool)

inception_output = Concatenate(axis=1, name='inception_output')([inception_1x1_conv,
                                                                 inception_3x3_conv,
                                                                 inception_5x5_conv,
                                                                 inception_3x3_pool_proj])

