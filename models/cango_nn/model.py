from keras.layers.core import Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import regularizers, optimizers, initializers

def create_model(input_dimension):

    reg_val = 0.001
    gnoise_val = 0.05
    dropout_val = 0.25
    learning_rate = 0.0001
    weight_init = initializers.random_normal(mean=0.0, stddev=0.05, seed=123)

    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=input_dimension,
                    kernel_initializer=weight_init,
                    bias_initializer='zero',
                    kernel_regularizer=regularizers.l2(reg_val),
                    activity_regularizer=regularizers.l2(reg_val)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(dropout_val))

    model.add(GaussianNoise(gnoise_val))
    model.add(Dense(256,
                    kernel_initializer=weight_init,
                    bias_initializer='zero',
                    kernel_regularizer=regularizers.l2(reg_val),
                    activity_regularizer=regularizers.l2(reg_val)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(dropout_val))

    model.add(GaussianNoise(gnoise_val))
    model.add(Dense(128,
                    kernel_initializer=weight_init,
                    bias_initializer='zero',
                    kernel_regularizer=regularizers.l2(reg_val),
                    activity_regularizer=regularizers.l2(reg_val)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(dropout_val))

    model.add(GaussianNoise(gnoise_val))
    model.add(Dense(64,
                    kernel_initializer=weight_init,
                    bias_initializer='zero',
                    kernel_regularizer=regularizers.l2(reg_val),
                    activity_regularizer=regularizers.l2(reg_val)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(dropout_val))

    model.add(GaussianNoise(gnoise_val))
    model.add(Dense(32,
                    kernel_initializer=weight_init,
                    bias_initializer='zero',
                    kernel_regularizer=regularizers.l2(reg_val),
                    activity_regularizer=regularizers.l2(reg_val)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(dropout_val))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.adam(lr=learning_rate),
                  metrics=['accuracy'])
    return model