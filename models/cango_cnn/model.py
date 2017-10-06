from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
from keras.models import Sequential
from keras.constraints import maxnorm
from keras import regularizers, optimizers, initializers
from keras import applications

def create_model():

    initializer = initializers.random_uniform(minval=0.00, maxval=1.00, seed=None)

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same',
                     input_shape=(1, 25, 25),
                     kernel_initializer=initializer,
                     bias_initializer='zero',
                     kernel_regularizer=regularizers.l2(0.001),
                     activity_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same',
                     kernel_regularizer=regularizers.l2(0.001),
                     activity_regularizer=regularizers.l2(0.001)
                     ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    # model.add(MaxPooling2D((2, 2), padding='same'))
    #
    # model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same',
    #                  kernel_regularizer=regularizers.l2(0.001),
    #                  activity_regularizer=regularizers.l2(0.001)
    #                  ))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same',
    #                  kernel_regularizer=regularizers.l2(0.001),
    #                  activity_regularizer=regularizers.l2(0.001)
    #                  ))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # # model.add(Dropout(0.25))
    #
    # model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Flatten())
    # model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001),
                    activity_regularizer=regularizers.l2(0.001)
                    ))
    # model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001),
                    activity_regularizer=regularizers.l2(0.001)
                    ))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=0.001),
                  metrics=['accuracy'])

    return model