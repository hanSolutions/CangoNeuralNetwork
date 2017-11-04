import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
from sklearn.metrics import roc_curve, auc, roc_auc_score


def train_val_acc(train_acc, val_acc, to_file=None, show=True):
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if to_file is not None:
        plt.savefig(to_file)

    if show:
        plt.show()


def train_val_loss(train_loss, val_loss, to_file=None, show=True):
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if to_file is not None:
        plt.savefig(to_file)

    if show:
        plt.show()


def roc_auc(y_true, y_score, to_file=None, show=True):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(to_file)
    if to_file is not None:
        plt.savefig(to_file)

    if show:
        plt.show()


def roc_auc_multi(y_true_arr, y_score_arr, label_arr, to_file=None, show=True):
    if len(y_true_arr) != len(y_score_arr) != len(label_arr):
        raise ValueError('array size not matching')

    plt.title('Receiver Operating Characteristic')
    count = len(y_true_arr)

    for i in range(0, count):
        fpr, tpr, _ = roc_curve(y_true_arr[i], y_score_arr[i])
        auc_roc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='%s = %0.4f' % (label_arr[i], auc_roc))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(to_file)
    if to_file is not None:
        plt.savefig(to_file)

    if show:
        plt.show()


def confusion_matrix(y_true, y_pred, to_file=None, show=True):
    conf_matrix = skmetrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    if to_file is not None:
        plt.savefig(to_file)

    if show:
        plt.show()