import os, datetime
import numpy as np
import logging
import datasets.cango as cango

from utils import plots, metrics
from common import logger, constants
from keras.models import model_from_json

log_dir_root = '../../logs'
out_dir_root = '../../outputs'

log = logging.getLogger(__name__)

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

    (x_train, y_train), (x_test, y_test) = cango.get_train_val_data(
        path='../../data/03_07_0_0_MaxMin01/clean_raw_pboc.csv',
        train_val_ratio=0.8, do_shuffle=False, do_smote=False)
    model_nn = get_model()

    # y_pred = p(class(x)=1)
    y_pred_test = model_nn.predict(x_test)
    y_pred_test = y_pred_test.ravel()
    proba_b_test = y_pred_test
    ones = np.ones(proba_b_test.shape).ravel()
    proba_g_test = ones - proba_b_test
    y_pred_test_out = [1 if e > 0.5 else 0 for e in y_pred_test]

    y_pred_train = model_nn.predict(x_train)
    y_pred_train = y_pred_train.ravel()
    y_pred_train_out = [1 if e > 0.5 else 0 for e in y_pred_train]

    # KS test score
    ks = metrics.ks_stat(proba_b_test, proba_g_test)
    log.info('ks score: {}'.format(ks))

    # PSI
    psi = metrics.psi(y_pred_train_out, y_pred_test_out)
    log.info('psi: {}'.format(psi))

    # AUC ROC
    auroc = metrics.auc_roc(y_true=y_test, y_score=y_pred_test)
    log.info("auc_roc score: {}".format(auroc))
    plots.roc_auc(y_true=y_test, y_score=y_pred_test,
                  to_file='{}/roc'.format(out_dir), show=True)

    # confusion matrix
    plots.confusion_matrix(y_true=y_test, y_pred=np.asarray(y_pred_test_out ),
                           to_file='{}/confusion'.format(out_dir),
                           show=True)