import os, datetime
import keras as ka
import matplotlib.pyplot as plt

from common import logger, constants
from datasets import cango
from keras.utils import plot_model
from models.cango_nn import model

log_dir_root = '../../logs'
out_dir_root = '../../outputs'

def main():
    log_dir, out_dir = init()

    (x_train, y_train), (x_val, y_val) = cango.get_train_val_data(
        path='../../data/03_07_0_0_Mean/clean_raw_pboc.csv',
        train_val_ratio=0.3, do_shuffle=True, do_smote=False, smote_min_ratio=0.3)

    # streams epoch results to a csv file
    csv_logger = ka.callbacks.CSVLogger('{}/{}_epoches.log'.format(log_dir, constants.APP_CANGO_NN))

    # checkpoint weight after each epoch if the validation loss decreased
    checkpointer = ka.callbacks.ModelCheckpoint(filepath='{}/{}_weights.hdf5'.format(out_dir, constants.APP_CANGO_NN),
                                                verbose=1,
                                                save_best_only=True)

    # stop training when a monitored quality has stopped improving
    early_stopping = ka.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    # Construct the model
    model_nn = model.create_model()

    # Train the model
    history = model_nn.fit(x_train, y_train, batch_size=100, nb_epoch=100,
                        verbose=0, validation_data=(x_val, y_val),
                        callbacks=[checkpointer, csv_logger, early_stopping])
    score = model_nn.evaluate(x_val, y_val, verbose=0)
    print('Validation score:', score[0])
    print('Validation accuracy:', score[1])

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{}/plt_acc'.format(out_dir))
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{}/plt_loss'.format(out_dir))
    plt.show()


    # Save the model
    json_string = model_nn.to_json()
    open('{}/model_architecture.json'.format(out_dir), 'w').write(json_string)
    plot_model(model_nn, to_file='{}/model.png'.format(out_dir))


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
    main()
