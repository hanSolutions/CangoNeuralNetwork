import sys
import keras as ka
import models.cango_nn_binclass.model as single_model

from utils import plots
from common import logger, constants, config
from datasets import cango_pboc
from keras.utils import plot_model
from models.cango_nn_multi_models.model import MultiModelsNeuralNetwork


def main(argv):
    config_file = argv[0]
    cfg = config.YamlParser(config_file)
    log_dir, out_dir = logger.init(log_dir=cfg.log_dir(),
                                   out_dir=cfg.out_dir(),
                                   level=cfg.log_level())

    (x_train, y_train), (x_val, y_val) = cango_pboc.get_train_val_data(
        path=cfg.train_data(), drop_columns=cfg.drop_columns(),
        train_val_ratio=cfg.train_val_ratio(),
        do_shuffle=cfg.do_shuffle(), do_smote=cfg.do_smote(), smote_ratio=cfg.smote_ratio())

    # streams epoch results to a csv file
    csv_logger = ka.callbacks.CSVLogger('{}/epoches.log'.format(log_dir))

    # checkpoint weight after each epoch if the validation loss decreased
    checkpointer = ka.callbacks.ModelCheckpoint(filepath='{}/weights.h5'.format(out_dir),
                                                verbose=1,
                                                save_best_only=True)

    # stop training when a monitored quality has stopped improving
    early_stopping = ka.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0, patience=10,
                                                verbose=1, mode='auto')

    # Construct the model
    input_dim = x_train.shape[1]
    mmnn = MultiModelsNeuralNetwork(input_dim)
    mmnn.set_reg_val(cfg.model_reg_val())
    mmnn.set_learning_rate(cfg.model_learning_rate())
    branch1 = single_model.create_model(input_dim,
                                        regularization_val=cfg.model_reg_val(),
                                        dropout_val=cfg.model_dropout_val(),
                                        learning_rate=cfg.model_learning_rate())
    mmnn.add_model(branch1)
    branch2 = single_model.create_model(input_dim,
                                        regularization_val=0.00001,
                                        dropout_val=cfg.model_dropout_val(),
                                        learning_rate=cfg.model_learning_rate())
    mmnn.add_model(branch2)
    model_nn = mmnn.create_model()

    # Train the model
    history = model_nn.fit([x_train, x_train], y_train,
                           batch_size=cfg.model_train_batch_size(),
                           epochs=cfg.model_train_epoches(),
                           verbose=0, validation_data=([x_val, x_val], y_val),
                           class_weight=cfg.model_class_weight(),
                           callbacks=[checkpointer, csv_logger, early_stopping])
    score = model_nn.evaluate([x_val, x_val], y_val, verbose=0)
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

    # save the model
    json_string = model_nn.to_json()
    open('{}/model_architecture.json'.format(out_dir), 'w').write(json_string)
    plot_model(model_nn, to_file='{}/model.png'.format(out_dir))


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) < 1:
        print("Expect input argument: config file path.")
        sys.exit()

    main(argv)
