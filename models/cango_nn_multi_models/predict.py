import os, sys, datetime
import numpy as np
import logging
import datasets.cango_pboc as cango

from utils import plots, metrics
from common import logger, config
from keras.models import model_from_json

log = logging.getLogger(__name__)


def get_model(model_path, weight_path):
    # Load the model architecture
    model = model_from_json(open('{}/model_architecture.json'.format(model_path)).read())
    # Load the model weights
    model.load_weights('{}/weights.h5'.format(weight_path))
    return model


def get_predict(model, data, batch_size):
    pred = model.predict(data, batch_size=batch_size)
    pred = pred.ravel()
    proba_c1 = pred
    ones = np.ones(shape=proba_c1.shape, dtype=np.float).ravel()
    proba_c0 = np.subtract(ones, proba_c1, dtype=np.float)
    predict_out = [1 if e > 0.5 else 0 for e in pred]
    return predict_out, proba_c0, proba_c1


def main(argv):
    config_file = argv[0]
    cfg = config.YamlParser(config_file)
    log_dir, out_dir = logger.init(log_dir=cfg.log_dir(),
                                   out_dir=cfg.out_dir(),
                                   level=cfg.log_level())

    (x_train, y_train), (x_val, y_val) = cango.get_train_val_data(
        path=cfg.train_data(), drop_columns=cfg.drop_columns(),
        train_val_ratio=cfg.train_val_ratio(), do_shuffle=True, do_smote=False)

    x_test, y_test = cango.get_test_data(
        path=cfg.test_data(), drop_columns=cfg.drop_columns())

    model_nn = get_model(cfg.out_dir(), cfg.out_dir())

    y_pred_val_out, proba_g_val, proba_b_val = get_predict(
        model=model_nn, data=[x_val, x_val], batch_size=100)
    y_pred_val_1 = np.count_nonzero(y_pred_val_out)
    y_pred_val_0 = len(y_pred_val_out) - y_pred_val_1
    log.debug('predict validation dataset distribution: 0 - {}, 1 - {}'.format(
        y_pred_val_0, y_pred_val_1
    ))

    y_pred_test_out, proba_g_test, proba_b_test = get_predict(
        model=model_nn, data=[x_test, x_test], batch_size=100)
    y_pred_test_1 = np.count_nonzero(y_pred_test_out)
    y_pred_test_0 = len(y_pred_test_out) - y_pred_test_1
    log.debug('predict test dataset distribution: 0 - {}, 1 - {}'.format(
        y_pred_test_0, y_pred_test_1
    ))

    # output
    np.savetxt('{}/predict.csv'.format(cfg.out_dir()),
               np.c_[y_pred_test_out, proba_g_test, proba_b_test],
               delimiter=',', header='Label, p_g, p_b',
               comments='', fmt='%d, %.6f, %.6f')
    np.savetxt('{}/predict_val.csv'.format(cfg.out_dir()),
               np.c_[y_pred_val_out, proba_g_val, proba_b_val],
               delimiter=',', header='Label, p_g, p_b',
               comments='', fmt='%d, %.6f, %.6f')

    # KS test score
    ks = metrics.ks_stat(proba_b_test, proba_g_test)
    log.info('ks val score: {}'.format(ks))

    # PSI
    psi = metrics.psi(y_pred_val_out, y_pred_test_out)
    log.info('psi: {}'.format(psi))

    # AUC ROC
    if y_test is not None:
        y_true_arr = [y_test, y_val]
        y_score_arr = [proba_b_test, proba_b_val]
        y_label_arr = ['AUC-test', 'AUC-val']
        plots.roc_auc_multi(y_true_arr=y_true_arr, y_score_arr=y_score_arr,
                            label_arr=y_label_arr,
                            to_file='{}/roc_all'.format(out_dir), show=True)
        # confusion matrix
        plots.confusion_matrix(y_true=y_test, y_pred=np.asarray(y_pred_test_out),
                               to_file='{}/confusion'.format(out_dir),
                               show=True)

    plots.confusion_matrix(y_true=y_val, y_pred=np.asarray(y_pred_val_out),
                           to_file='{}/confusion1'.format(out_dir),
                           show=True)


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) < 1:
        print("Expect input argument: config file path.")
        sys.exit()

    main(argv)