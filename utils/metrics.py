import numpy as np

from scipy import stats
from sklearn import metrics


def ks_scipy(seq1, seq2):
    '''
    :param seq1: array, shape=[n_samples]
    :param seq2: array, shape=[n_samples]
    :return:
    KS: float, KS statistic
    pval: two-tailed p-value
    '''
    ks, pval = stats.ks_2samp(seq1, seq2)
    return ks, pval


def ks_stat(seq1, seq2):
    '''
    :param seq1: array, shape=[n_samples]
    :param seq2: array, shape=[n_samples]
    :return: float, KS statistic
    '''
    d1 = np.sort(seq1)
    d2 = np.sort(seq2)
    n1 = d1.shape[0]
    n2 = d2.shape[0]
    data_all = np.concatenate([d1, d2])
    cdf1 = np.searchsorted(d1, data_all, side='right') / (1.0*n1)
    cdf2 = np.searchsorted(d2, data_all, side='right') / (1.0*n2)
    d = np.max(np.absolute(cdf1 - cdf2))
    return d


def psi(expected, actual):
    '''
    :param expected: 1D array, elements are either 0 or 1
    :param actual: 1D array, elements are either 0 or 1
    :return: float, Population Stability Index (PSI)
    '''
    exp_len = len(expected)
    act_len = len(actual)

    nb_exp = np.count_nonzero(expected)
    ng_exp = exp_len - nb_exp

    nb_act = np.count_nonzero(actual)
    ng_act = act_len - nb_act

    ratio_g_exp = ng_exp / exp_len
    ratio_b_exp = nb_exp / exp_len
    ratio_g_act = ng_act / act_len
    ratio_b_act = nb_act / act_len

    g = (ratio_g_act - ratio_g_exp) * np.log(ratio_g_act / ratio_g_exp)
    b = (ratio_b_act - ratio_b_exp) * np.log(ratio_b_act / ratio_b_exp)

    return g + b


def auc_roc(y_true, y_score):
    '''
    :param y_true: array, shape=[n_samples]
    :param y_score: array, shape=[n_samples]
    :return: float, the area under the ROC-curve
    '''
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    _auc = metrics.auc(fpr, tpr)
    return _auc