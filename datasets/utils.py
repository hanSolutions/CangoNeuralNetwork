import logging
import numpy as np
import pandas_ml as pdml
import common.constants as c

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)


def flat_to_one_hot(labels, categorical=True):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y


def shuffle(data):
    return np.random.shuffle(data)


def data_split2(data, split_ratio):
    data_size = len(data)
    split_size = int(data_size * (1 - split_ratio))

    split_left = data[:split_size]
    split_right = data[split_size:]
    return split_left, split_right


def data_split3(data, ratio1, ratio2):
    split_left, split_mid, split_right = \
        np.split(data, [int(ratio1*len(data)), int(ratio2*len(data))])
    return split_left, split_mid, split_right


def reshape(data, reshape_size):
    d = reshape_size[0]
    w = reshape_size[1]
    l = reshape_size[2]

    resize_td = np.resize(data, (data.shape[0], d * w * l))
    data_reshaped = resize_td.reshape(resize_td.shape[0], d, w, l)
    return data_reshaped


def smote(data, labels, ratio=0.5):
    # https://www.jair.org/media/953/live-953-2037-jair.pdf
    mdf = pdml.ModelFrame(data, target=labels)
    ratio = int(data.shape[0] * ratio)
    ratio_dict = {1: ratio}

    sampler = mdf.imbalance.over_sampling.SMOTE(ratio_dict, random_state=c.random_seed)
    sampled = mdf.fit_sample(sampler)

    num_zero = sampled.target.value_counts()[0]
    num_one = sampled.target.value_counts()[1]

    label_sampled = sampled.target.values
    sampled.drop('.target', axis=1, inplace=True)
    data_sampled = sampled.values
    return data_sampled, label_sampled, num_zero, num_one

