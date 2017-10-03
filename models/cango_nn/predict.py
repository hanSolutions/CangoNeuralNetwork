import os, datetime
import numpy as np
import datasets.cango as cango

from utils import plots
from common import logger, constants
from keras.models import model_from_json

log_dir_root = '../../logs'
out_dir_root = '../../outputs'


def get_model():
    # Load the model architecture
    model = model_from_json(open('../../outputs/cango_nn/model_architecture.json').read())
    # Load the model weights
    model.load_weights('../../outputs/cango_nn/cango_nn_weights.hdf5')
    return model


def init():
    dtstr = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_dir = os.path.join(log_dir_root,
                           "{}_{}".format(constants.APP_CANGO_NN, dtstr))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    out_dir = os.path.join(out_dir_root,
                           constants.APP_CANGO_NN)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.init_log('DEBUG')

    return log_dir, out_dir


if __name__ == '__main__':
    log_dir, out_dir = init()

    (_, _), (x_test, y_test) = cango.get_train_val_data(
        path='../../data/03_07_0_0_MaxMin/clean_raw_pboc.csv',
        train_val_ratio=0.3, do_shuffle=False, do_smote=False)
    model_nn = get_model()

    y_pred = model_nn.predict(x_test)
    dfd = np.argmax(y_pred, axis=1)

    # t = roc_auc_score(y_test, y_pred)
    # print(t)
    # loss = K.mean(K.binary_crossentropy(y_test, y_pred), axis=-1)
    # loss = np.mean(log_loss(y_test, y_pred))
    # loss = np.mean(np.power(y_test - y_pred, 2), axis=1)
    y_pred = y_pred.ravel()

    # roc
    plots.roc_auc(y_true=y_test, y_score=y_pred,
                  to_file='{}/roc'.format(out_dir), show=True)

    # confusion matrix
    threshold = 0.5
    y_pred = [1 if e > threshold else 0 for e in y_pred]
    plots.confusion_matrix(y_true=y_test, y_pred=np.asarray(y_pred),
                           to_file='{}/confusion'.format(out_dir),
                           show=True)