import logging
import numpy as np
import pandas as pd
import pandas_ml as pdml
import common.constants as c

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
    input_data.drop(c.DAT_COL_PBOC_SPOUSE, axis=1, inplace=True)

    dataset = input_data.values
    num_rows = input_data.shape[0]
    num_cols = input_data.shape[1]

    # shuffle
    if do_shuffle:
        log.info('Shuffling data...')
        dataset = shuffle(dataset)

    # labels one-hot
    labels = dataset[:, num_cols - 1]
    labels = flat_to_one_hot(labels, categorical=False)
    dataset = np.delete(dataset, -1, axis=1)

    # split train validation set
    validation_size = int(num_rows * train_val_ratio)

    log.info('Training data set: {}, Validation data set: {}'.format(
             num_rows - validation_size,
             validation_size))

    validation_dataset = dataset[:validation_size]
    validation_labels = labels[:validation_size]
    log.debug('Validation label distribution: 0 - {}, 1 - {}'.format(
        validation_labels.shape[0] - np.count_nonzero(validation_labels),
        np.count_nonzero(validation_labels)))

    train_dataset = dataset[validation_size:]
    train_labels = labels[validation_size:]
    log.debug('Training label distribution: 0 - {}, 1 - {}'.format(
        train_labels.shape[0] - np.count_nonzero(train_labels),
        np.count_nonzero(train_labels)))

    # Synthetic Minority Over-sampling (SMOTE) only on train dataset
    # https://www.jair.org/media/953/live-953-2037-jair.pdf
    if do_smote:
        log.info('Processing SMOTE on training dataset..')
        mdf = pdml.ModelFrame(train_dataset,
                              target=train_labels)

        ratio = int(input_data.shape[0] * smote_min_ratio)
        ratio_dict = {1: ratio}

        sampler = mdf.imbalance.over_sampling.SMOTE(ratio_dict, random_state=c.random_seed)
        sampled = mdf.fit_sample(sampler)
        log.debug('Train dataset SMOTE result: total - {}, 0 - {}, 1 - {}'.format(
            sampled.shape[0],
            sampled.target.value_counts()[0],
            sampled.target.value_counts()[1]
        ))
        train_labels = sampled.target.values
        sampled.drop('.target', axis=1, inplace=True)
        train_dataset = sampled.values
        # input_data = sampled
        # input_data.drop('.target', axis=1, inplace=True)

    # reshape 1-D array to 2-D
    if do_reshape:
        # resize ndarray(*, 93) to ndarray(*, 100) with zero padding

        d = reshape_size[0]
        w = reshape_size[1]
        l = reshape_size[2]

        resize_td = np.resize(train_dataset, (train_dataset.shape[0], d * w * l))
        train_dataset = resize_td.reshape(resize_td.shape[0], d, w, l)
        resize_vd = np.resize(validation_dataset, (validation_dataset.shape[0], d * w * l))
        validation_dataset = resize_vd.reshape(resize_vd.shape[0], d, w, l)

    return (train_dataset, train_labels), (validation_dataset, validation_labels)



def get_test_data(path=None,
                       do_shuffle=False,
                       do_reshape=False,
                       reshape_size=[1, 25, 25]):
    if path is None:
        raise ValueError('Undefined input file path')

    log.info("Loading data from '{}'".format(path))
    input_data = pd.read_csv(path)

    # drop 'id' column from training set
    input_data.drop(c.DAT_COL_PBOC_SPOUSE, axis=1, inplace=True)

    dataset = input_data.values
    num_rows = input_data.shape[0]
    num_cols = input_data.shape[1]
    log.debug('total numbers of test data: {}'.format(num_rows))

    # shuffle
    if do_shuffle:
        log.info('Shuffling data...')
        dataset = shuffle(dataset)

    # labels one-hot
    labels = dataset[:, num_cols - 1]
    labels = flat_to_one_hot(labels, categorical=False)
    dataset = np.delete(dataset, -1, axis=1)

    # reshape 1-D array to 2-D
    if do_reshape:
        # resize ndarray(*, 93) to ndarray(*, 100) with zero padding

        d = reshape_size[0]
        w = reshape_size[1]
        l = reshape_size[2]

        resize_td = np.resize(dataset, (dataset.shape[0], d * w * l))
        dataset = resize_td.reshape(resize_td.shape[0], d, w, l)

    return (dataset, labels)
