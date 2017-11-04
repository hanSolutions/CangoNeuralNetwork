import logging
import numpy as np
import pandas as pd
import common.constants as c
import datasets.utils as dat_utils

log = logging.getLogger(__name__)


def get_train_val_data(path=None,
                       drop_columns=None,
                       categorical_labels=False,
                       train_val_ratio=0.2,
                       do_shuffle=False,
                       do_smote=False,
                       smote_ratio=0.2,
                       do_reshape=False,
                       reshape_size=[1, 25, 25]):

    if path is None:
        raise ValueError('Undefined input file path')

    log.info("Loading data from '{}'".format(path))
    input_data = pd.read_csv(path)

    # drop columns from training set
    if drop_columns is not None:
        for col in drop_columns:
            input_data.drop(col, axis=1, inplace=True)

    dataset = input_data.values
    num_rows = input_data.shape[0]
    num_cols = input_data.shape[1]

    # shuffle
    if do_shuffle:
        log.info('Shuffling data...')
        dat_utils.shuffle(dataset)

    # split train validation set
    labels = dataset[:, num_cols - 1]
    # one hot labels
    labels = dat_utils.flat_to_one_hot(labels, categorical=categorical_labels)
    dataset = np.delete(dataset, -1, axis=1)

    train_dataset, validation_dataset = dat_utils.data_split2(data=dataset, split_ratio=train_val_ratio)
    train_labels, validation_labels = dat_utils.data_split2(data=labels, split_ratio=train_val_ratio)

    log.debug('Training label distribution: 0 - {}, 1 - {}'.format(
        train_labels.shape[0] - np.count_nonzero(train_labels),
        np.count_nonzero(train_labels)))

    log.debug('Validation label distribution: 0 - {}, 1 - {}'.format(
        validation_labels.shape[0] - np.count_nonzero(validation_labels),
        np.count_nonzero(validation_labels)))

    # Synthetic Minority Over-sampling (SMOTE) only on train dataset
    if do_smote:
        train_dataset, train_labels, num_zero, num_one = \
            dat_utils.smote(data=train_dataset, labels=train_labels, ratio=smote_ratio)
        log.debug('SMOTE result: total - {}, 0 - {}, 1 - {}'.format(
            len(train_dataset), num_zero, num_one))

    # reshape 1-D array to 2-D
    if do_reshape:
        train_dataset = dat_utils.reshape(data=train_dataset, reshape_size=reshape_size)
        validation_dataset = dat_utils.reshape(data=validation_dataset,
                                               reshape_size=reshape_size)

    return (train_dataset, train_labels), (validation_dataset, validation_labels)


def get_train_val_test_data(path=None,
                       drop_columns=None,
                       categorical_labels=False,
                       train_val_ratio=0.2,
                       do_shuffle=False,
                       do_smote=False,
                       smote_ratio=0.2,
                       do_reshape=False,
                       reshape_size=[1, 25, 25]):

    if path is None:
        raise ValueError('Undefined input file path')

    log.info("Loading data from '{}'".format(path))
    input_data = pd.read_csv(path)

    # drop columns from training set
    if drop_columns is not None:
        for col in drop_columns:
            input_data.drop(col, axis=1, inplace=True)

    dataset = input_data.values
    num_rows = input_data.shape[0]
    num_cols = input_data.shape[1]

    # shuffle
    if do_shuffle:
        log.info('Shuffling data...')
        dat_utils.shuffle(dataset)

    # split train validation set
    labels = dataset[:, num_cols - 1]
    # one hot labels
    labels = dat_utils.flat_to_one_hot(labels, categorical=categorical_labels)
    dataset = np.delete(dataset, -1, axis=1)

    # train_dataset, validation_dataset = dat_utils.data_split2(data=dataset, split_ratio=train_val_ratio)
    # train_labels, validation_labels = dat_utils.data_split2(data=labels, split_ratio=train_val_ratio)

    train_dataset, validation_dataset, test_dataset =\
        dat_utils.data_split3(data=dataset, ratio1=.6, ratio2=.8)
    train_labels, validation_labels, test_labels =\
        dat_utils.data_split3(data=labels, ratio1=.6, ratio2=.8)

    log.debug('Training label distribution: 0 - {}, 1 - {}'.format(
        train_labels.shape[0] - np.count_nonzero(train_labels),
        np.count_nonzero(train_labels)))

    log.debug('Validation label distribution: 0 - {}, 1 - {}'.format(
        validation_labels.shape[0] - np.count_nonzero(validation_labels),
        np.count_nonzero(validation_labels)))

    # Synthetic Minority Over-sampling (SMOTE) only on train dataset
    if do_smote:
        train_dataset, train_labels, num_zero, num_one = \
            dat_utils.smote(data=train_dataset, labels=train_labels, ratio=smote_ratio)
        log.debug('SMOTE result: total - {}, 0 - {}, 1 - {}'.format(
            len(train_dataset), num_zero, num_one))

    # reshape 1-D array to 2-D
    if do_reshape:
        train_dataset = dat_utils.reshape(data=train_dataset, reshape_size=reshape_size)
        validation_dataset = dat_utils.reshape(data=validation_dataset,
                                               reshape_size=reshape_size)

    return (train_dataset, train_labels),\
           (validation_dataset, validation_labels),\
           (test_dataset, test_labels)


def get_test_data(path=None,
                  drop_columns=None,
                  categorical_labels=False,
                  do_reshape=False,
                  reshape_size=[1, 25, 25]):
    if path is None:
        raise ValueError('Undefined input file path')

    log.info("Loading data from '{}'".format(path))
    input_data = pd.read_csv(path)

    # drop 'id' column from training set
    if drop_columns is not None:
        for col in drop_columns:
            input_data.drop(col, axis=1, inplace=True)

    dataset = input_data.values
    num_rows = input_data.shape[0]
    num_cols = input_data.shape[1]
    log.debug('total numbers of test data: {}'.format(num_rows))

    # shuffle
    dat_utils.shuffle(dataset)

    # labels one-hot
    labels = dataset[:, num_cols - 1]
    labels = dat_utils.flat_to_one_hot(labels, categorical=categorical_labels)
    dataset = np.delete(dataset, -1, axis=1)

    # reshape 1-D array to 2-D
    if do_reshape:
        dataset = dat_utils.reshape(data=dataset, reshape_size=reshape_size)

    return dataset, labels

