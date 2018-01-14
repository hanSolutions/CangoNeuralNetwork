import os, sys, datetime
import numpy as np
import pandas as pd
import logging
import datasets.cango_pboc as cango
import matplotlib.pyplot as plt

from utils import plots, metrics, psi3
from common import logger, config
from keras.models import model_from_json

log = logging.getLogger(__name__)


def get_model(model_path, weight_path):
    # Load the model architecture
    model = model_from_json(open('{}/model_architecture.json'.format(model_path)).read())
    # Load the model weights
    model.load_weights('{}/weights.h5'.format(weight_path))
    return model


def get_predict(model, data, batch_size, cutoff):
    pred = model.predict(data, batch_size=batch_size)
    pred = pred.ravel()
    proba_c1 = pred
    ones = np.ones(shape=proba_c1.shape, dtype=np.float).ravel()
    proba_c0 = np.subtract(ones, proba_c1, dtype=np.float)
    predict_out = [1 if e > 1.0 - cutoff else 0 for e in pred]
    return predict_out, proba_c0, proba_c1


def main(argv):
    config_file = argv[0]

    cfg = config.YamlParser(config_file)
    log_dir, out_dir = logger.init(log_dir=cfg.log_dir(),
                                   out_dir=cfg.out_dir(),
                                   level=cfg.log_level())

    if cfg.one_filer():
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = cango.get_train_val_test_data(
            path=cfg.train_data(), drop_columns=cfg.drop_columns(),
            train_val_ratio=cfg.train_val_ratio(),
            do_shuffle=cfg.do_shuffle(), do_smote=False, smote_ratio=cfg.smote_ratio())
    else:
        (x_train, y_train), (x_val, y_val) = cango.get_train_val_data(
            path=cfg.train_data(), drop_columns=cfg.drop_columns(),
            train_val_ratio=cfg.train_val_ratio(),
            do_shuffle=cfg.do_shuffle(), do_smote=False, smote_ratio=cfg.smote_ratio())

        x_test, y_test = cango.get_test_data(
            path=cfg.test_data(), drop_columns=cfg.drop_columns())

    model_nn = get_model(cfg.out_dir(), cfg.out_dir())

    y_pred_train_out, proba_g_train, proba_b_train = get_predict(
        model=model_nn, data=x_train, batch_size=100, cutoff=cfg.cutoff())
    y_pred_val_1 = np.count_nonzero(y_pred_train_out)
    y_pred_val_0 = len(y_pred_train_out) - y_pred_val_1
    log.debug('predict train dataset distribution: 0 - {}, 1 - {}'.format(
        y_pred_val_0, y_pred_val_1
    ))

    y_pred_val_out, proba_g_val, proba_b_val = get_predict(
        model=model_nn, data=x_val, batch_size=100, cutoff=cfg.cutoff())
    y_pred_val_1 = np.count_nonzero(y_pred_val_out)
    y_pred_val_0 = len(y_pred_val_out) - y_pred_val_1
    log.debug('predict validation dataset distribution: 0 - {}, 1 - {}'.format(
        y_pred_val_0, y_pred_val_1
    ))

    y_pred_test_out, proba_g_test, proba_b_test = get_predict(
        model=model_nn, data=x_test, batch_size=100, cutoff=cfg.cutoff())
    y_pred_test_1 = np.count_nonzero(y_pred_test_out)
    y_pred_test_0 = len(y_pred_test_out) - y_pred_test_1
    log.debug('predict test dataset distribution: 0 - {}, 1 - {}'.format(
        y_pred_test_0, y_pred_test_1
    ))

    df_test = None
    df_val = None
    # output
    if y_test is not None:
        np.savetxt('{}/predict_test.csv'.format(cfg.out_dir()),
                   np.c_[y_test, y_pred_test_out, proba_g_test, proba_b_test],
                   delimiter=',', header='CG_Label, Label, p_g, p_b',
                   comments='', fmt='%d, %d, %.6f, %.6f')
        df_test = pd.DataFrame({
            'CG_Label': y_test,
            'Label': y_pred_test_out,
            'p_g': proba_g_test,
            'p_b': proba_b_test
        })
        bins_test, c0_test, c1_test = metrics.cals_KS_bins(df_test, 'p_b', 'Label')
        np.savetxt('{}/predict_bin_test.csv'.format(cfg.out_dir()),
                   np.c_[bins_test, c0_test, c1_test],
                   delimiter=',', header='p_b, n_g_label, n_b_label',
                   comments='', fmt='%.1f, %d, %d')
    else:
        np.savetxt('{}/predict_bin_test.csv'.format(cfg.out_dir()),
                   np.c_[y_pred_test_out, proba_g_test, proba_b_test],
                   delimiter=',', header='Label, p_g, p_b',
                   comments='', fmt='%d, %.6f, %.6f')

    np.savetxt('{}/predict_val.csv'.format(cfg.out_dir()),
               np.c_[y_val, y_pred_val_out, proba_g_val, proba_b_val],
               delimiter=',', header='CG_Label, Label, p_g, p_b',
               comments='', fmt='%d, %d, %.6f, %.6f')

    df_val = pd.DataFrame({
        'CG_Label': y_val,
        'Label': y_pred_val_out,
        'p_g': proba_g_val,
        'p_b': proba_b_val
    })
    bins_val, c0_val, c1_val = metrics.cals_KS_bins(df_val, 'p_b', 'CG_Label')
    np.savetxt('{}/predict_bin_val.csv'.format(cfg.out_dir()),
               np.c_[bins_val, c0_val, c1_val],
               delimiter=',', header='p_b, n_g_label, n_b_Label',
               comments='', fmt='%.1f, %d, %d')

    # KS test score
    ks_val = metrics.calc_KS_AR(df_val, 'p_g', 'CG_Label')
    ks_val_value = np.max(np.subtract(ks_val[1]['badCumPer'].values, ks_val[1]['goodCumPer'].values))
    log.info('ks val score: {}'.format(ks_val_value))
    ks_test = metrics.calc_KS_AR(df_test, 'p_g', 'CG_Label')
    ks_test_value = np.max(np.subtract(ks_test[1]['badCumPer'].values, ks_test[1]['goodCumPer'].values))
    log.info('ks test score: {}'.format(ks_test_value))

    plt.figure(figsize=(14, 10), dpi=80, facecolor='w')
    plt.plot(ks_val[1]['p_g'], ks_val[1]['goodCumPer'], lw=2, alpha=0.8, label='Good Percent -val')
    plt.plot(ks_test[1]['p_g'], ks_test[1]['goodCumPer'], lw=2, alpha=0.8, label='Good Percent -test')
    plt.plot(ks_val[1]['p_g'], ks_val[1]['badCumPer'], lw=2, alpha=0.8, label='Bad Percent- val')
    plt.plot(ks_test[1]['p_g'], ks_test[1]['badCumPer'], lw=2, alpha=0.8, label='Bad Percent -test')
    #plt.xticks(list(train_ks[1]['goodCumPer'].index), list(train_ks[1]['train_proba'].unique()), rotation=90)
    plt.title('K-S curve', fontsize=18)
    plt.xlabel('p_b', fontsize=14)
    plt.ylabel('good/bad percent', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(b=True, ls=':')
    plt.savefig('{}/ks'.format(cfg.out_dir()))
    plt.show()

    # PSI
    psiCalc = psi3.PSI()
    psi_val = psiCalc.calcPSI(y_pred_test_out, proba_b_test, y_pred_val_out, proba_b_val)
    log.info('PSI (p_b): {}'.format(psi_val))
    psi_val = psiCalc.calcPSI(y_pred_test_out, proba_g_test, y_pred_val_out, proba_g_val)
    log.info('PSI (p_g): {}'.format(psi_val))

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
                               to_file='{}/confusion_test'.format(out_dir),
                               show=True)

    plots.confusion_matrix(y_true=y_val, y_pred=np.asarray(y_pred_val_out),
                           to_file='{}/confusion_val'.format(out_dir),
                           show=True)


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) < 1:
        print("Expect input argument: config file path.")
        sys.exit()

    main(argv)
