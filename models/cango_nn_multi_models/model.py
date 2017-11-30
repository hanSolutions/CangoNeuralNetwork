from keras.layers.core import Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers import Concatenate, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras import regularizers, optimizers, initializers

import common.constants as c

DEFAULT_REG_VAL = 0.0001
DEFAULT_DROPOUT_VAL = 0.5
DEFAULT_GUASSIAN_NOISE_VAL = 0.25
DEFAULT_LEARNING_RATE = 0.0001


class MultiModelsNeuralNetwork(object):

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.reg_val = DEFAULT_REG_VAL
        self.dropout_val = DEFAULT_DROPOUT_VAL
        self.gaussian_noise_val = DEFAULT_GUASSIAN_NOISE_VAL
        self.learning_rate = DEFAULT_LEARNING_RATE
        self.models = []

    def add_model(self, model):
        self.models.append(model)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_reg_val(self, reg_val):
        self.reg_val = reg_val

    def create_model(self):
        # branch1 = self.__create_sub_model()
        # branch2 = self.__create_sub_model()
        # branch3 = self.__create_sub_model()
        # m4 = self.__create_sub_model()

        model = Sequential()
        model.add(Merge(self.models, mode='concat'))

        model.add(Dense(8,
                        kernel_initializer=initializers.random_normal(mean=0.01, stddev=0.05, seed=c.random_seed),
                        bias_initializer='zero',
                        kernel_regularizer=regularizers.l2(self.reg_val),
                        activity_regularizer=regularizers.l2(self.reg_val)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Dense(1, init='normal', activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.nadam(lr=self.learning_rate),
                           metrics=['accuracy'])
        return model

