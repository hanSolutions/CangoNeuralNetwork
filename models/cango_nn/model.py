from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import regularizers, optimizers, initializers

def create_model():

    # create model
    model = Sequential()
    model.add(Dense(512, input_dim=603,
                    kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
                    kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(256,
                    kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
                    kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    # model.add(Dense(128,
    #                 kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=123),
    #                 kernel_regularizer=regularizers.l2(0.01),
    #                 activity_regularizer=regularizers.l2(0.01)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Dense(2, kernel_initializer='normal', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.adam(lr=0.0001),
                  metrics=['accuracy'])
    return model