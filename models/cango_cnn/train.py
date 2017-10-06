import os, datetime
import keras as ka
import numpy as np
import pandas as pd

from utils import plots
from common import logger, constants
from datasets import cango
from keras.utils import plot_model
from models.cango_cnn import model

log_dir_root = '../../logs'
out_dir_root = '../../outputs'


def main():
    log_dir, out_dir = init()

    (x_train, y_train), (x_val, y_val) = cango.get_train_val_data(
        path='../../data/03_07_0_0_MaxMin01/clean_raw_pboc.csv',
        train_val_ratio=0.3, do_shuffle=True, do_smote=True, smote_min_ratio=0.9,
        do_reshape=True
    )

    # streams epoch results to a csv file
    csv_logger = ka.callbacks.CSVLogger('{}/{}_epoches.log'.format(log_dir, constants.APP_CANGO_CNN))

    # checkpoint weight after each epoch if the validation loss decreased
    checkpointer = ka.callbacks.ModelCheckpoint(filepath='{}/{}_weights.hdf5'.format(out_dir, constants.APP_CANGO_CNN),
                                                verbose=1,
                                                save_best_only=True)

    # stop training when a monitored quality has stopped improving
    early_stopping = ka.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    # Construct the model
    input_dim = x_train.shape[1]
    model_nn = model.create_model()

    # if os.path.exists('../../outputs/cango_nn/cango_nn_weights.hdf5'):
    #     model_nn.load_weights('../../outputs/cango_nn/cango_nn_weights.hdf5')

    # Train the model
    history = model_nn.fit(x_train, y_train, batch_size=100, epochs=100,
                        verbose=1, validation_data=(x_val, y_val),
                        callbacks=[checkpointer, csv_logger, early_stopping])
    score = model_nn.evaluate(x_val, y_val, verbose=0)
    print('Validation score:', score[0])
    print('Validation accuracy:', score[1])

    # summarize history for accuracy
    plots.train_val_acc(train_acc=history.history['acc'],
                        val_acc=history.history['val_acc'],
                        to_file='{}/plt_acc'.format(out_dir),
                        show=True)

    # summarize history for loss
    plots.train_val_loss(train_loss=history.history['loss'],
                         val_loss=history.history['val_loss'],
                         to_file='{}/plt_loss'.format(out_dir),
                         show=True)

    predictions = model_nn.predict(x_val)
    # mse = np.mean(np.power(x_val - predictions, 2), axis=1)
    # error_df = pd.DataFrame({'reconstruction_error': mse,
    #                          'true_class': y_val})

    # Save the model
    json_string = model_nn.to_json()
    open('{}/model_architecture.json'.format(out_dir), 'w').write(json_string)
    plot_model(model_nn, to_file='{}/model.png'.format(out_dir))


def init():
    dtstr = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_dir = os.path.join(log_dir_root,
                           "{}_{}".format(constants.APP_CANGO_CNN, dtstr))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    out_dir = os.path.join(out_dir_root,
                           constants.APP_CANGO_CNN)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.init_log('DEBUG')

    return log_dir, out_dir


if __name__ == '__main__':
    main()
