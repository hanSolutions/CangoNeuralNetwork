import logging
import numpy as np
import pandas as pd
import pandas_ml as pdml
import common.constants as const

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

log = logging.getLogger(__name__)


def flat_to_one_hot(labels, categorical=True):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y


def get_train_val_data(path=None,
                       train_val_ratio=0.2,
                       do_shuffle=False,
                       do_smote=False,
                       smote_min_ratio=0.2,
                       do_reshape=False,
                       reshape_size=[1, 25, 25]):

    if path is None:
        raise ValueError('Undefined input file path')

    log.info("Loading data from '{}'".format(path))
    input_data = pd.read_csv(path)

    # drop 'id' column from training set
    input_data.drop(const.DAT_COL_PBOC_SPOUSE, axis=1, inplace=True)


    # shuffle
    if do_shuffle:
        log.info("Shuffling data...")
        input_data = shuffle(input_data)

    # Synthetic Minority Over-sampling
    if do_smote:
        log.info("Data preprocess: SMOTE...")
        mdf = pdml.ModelFrame(input_data.to_dict(orient='list'),
                              target=input_data[const.DAT_COL_LABEL].values)

        ratio = int(input_data.shape[0] * smote_min_ratio)
        ratio_dict = {1: ratio}

        sampler = mdf.imbalance.over_sampling.SMOTE(ratio_dict, random_state=123)
        sampled = mdf.fit_sample(sampler)
        log.debug("Data preprocess: SMOTE result: total - {}, 0 - {}, 1 - {}".format(
            sampled.shape[0],
            sampled.target.value_counts()[0],
            sampled.target.value_counts()[1]
        ))
        input_data = sampled
        input_data.drop('.target', axis=1, inplace=True)

    # labels one-hot
    labels = input_data[const.DAT_COL_LABEL].values
    labels = flat_to_one_hot(labels, categorical=True)
    input_data.drop(const.DAT_COL_LABEL, axis=1, inplace=True)

    # split train validation set
    dataset = input_data.values
    total = input_data.shape[0]
    validation_size = int(total * train_val_ratio)

    validation_dataset = dataset[:validation_size]
    validation_labels = labels[:validation_size]

    train_dataset = dataset[validation_size:]
    train_labels = labels[validation_size:]

    # reshape 1-D array to 2-D
    if do_reshape:
        # resize ndarray(*, 93) to ndarray(*, 100) with zero padding
        resize_td = np.resize(train_dataset, (train_dataset.shape[0], 100))
        train_dataset = resize_td.reshape(resize_td.shape[0], 1, 10, 10)
        resize_vd = np.resize(validation_dataset, (validation_dataset.shape[0], 100))
        validation_dataset = resize_vd.reshape(resize_vd.shape[0], 1, 10, 10)

    return (train_dataset, train_labels), (validation_dataset, validation_labels)
