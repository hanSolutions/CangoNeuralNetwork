from keras.layers.core import Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras import regularizers, optimizers, initializers

import common.constants as c


def create_model(input_dimension,
                 regularization_val=0.0001,
                 dropout_val=0.5,
                 gaussian_noise_val=0.0,
                 learning_rate=0.0001
                 ):

    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=input_dimension,
                    kernel_initializer=initializers.random_normal(mean=0.01, stddev=0.05, seed=c.random_seed),
                    bias_initializer='zero',
                    kernel_regularizer=regularizers.l2(regularization_val),
                    activity_regularizer=regularizers.l2(regularization_val)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(dropout_val))

    # model.add(GaussianNoise(gaussian_noise_val))
    model.add(Dense(256,
                    kernel_initializer=initializers.random_normal(mean=0.01, stddev=0.05, seed=c.random_seed),
                    bias_initializer='zero',
                    kernel_regularizer=regularizers.l2(regularization_val),
                    activity_regularizer=regularizers.l2(regularization_val)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(dropout_val))

    # model.add(GaussianNoise(gaussian_noise_val))
    model.add(Dense(128,
                    kernel_initializer=initializers.random_normal(mean=0.01, stddev=0.05, seed=c.random_seed),
                    bias_initializer='zero',
                    kernel_regularizer=regularizers.l2(regularization_val),
                    activity_regularizer=regularizers.l2(regularization_val)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(dropout_val))

    # model.add(GaussianNoise(gaussian_noise_val))
    model.add(Dense(64,
                    kernel_initializer=initializers.random_normal(mean=0.01, stddev=0.05, seed=c.random_seed),
                    bias_initializer='zero',
                    kernel_regularizer=regularizers.l2(regularization_val),
                    activity_regularizer=regularizers.l2(regularization_val)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(dropout_val))

    # model.add(GaussianNoise(gaussian_noise_val))
    model.add(Dense(32,
                    kernel_initializer=initializers.random_normal(mean=0.01, stddev=0.05, seed=c.random_seed),
                    bias_initializer='zero',
                    kernel_regularizer=regularizers.l2(regularization_val),
                    activity_regularizer=regularizers.l2(regularization_val)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(dropout_val))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.nadam(lr=learning_rate),
                  metrics=['accuracy'])
    return model
