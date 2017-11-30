import sys
import os
import keras as ka
import numpy as np
import models.cango_nn_binclass.model as single_model
from common import logger, constants, config
from datasets import cango_pboc
from sklearn.model_selection import StratifiedKFold
from models.cango_nn_multi_models.model import MultiModelsNeuralNetwork


def main(argv):
    config_file = argv[0]
    cfg = config.YamlParser(config_file)
    log_dir, out_dir = logger.init(log_dir=cfg.log_dir(),
                                   out_dir=cfg.out_dir(),
                                   level=cfg.log_level())
    weight_path = '{}/weights.h5'.format(out_dir)

    (X, Y), (_, _) = cango_pboc.get_train_val_data(
        path=cfg.train_data(), drop_columns=cfg.drop_columns(),
        train_val_ratio=cfg.train_val_ratio(),
        do_shuffle=cfg.do_shuffle(), do_smote=cfg.do_smote(), smote_ratio=cfg.smote_ratio())

    kfold = StratifiedKFold(n_splits=10, shuffle=True,
                            random_state=constants.random_seed)

    checkpointer = ka.callbacks.ModelCheckpoint(filepath=weight_path,
                                                verbose=1,
                                                save_best_only=True)

    # Construct the model
    input_dim = X.shape[1]
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

    cvscores = []
    for train_index, test_index in kfold.split(X, Y):


        if os.path.exists(weight_path):
            model_nn.load_weights(weight_path)

        early_stopping = ka.callbacks.EarlyStopping(monitor='val_loss',
                                                    min_delta=0, patience=5,
                                                    verbose=1, mode='auto')

        model_nn.fit([X[train_index], X[train_index]],
                     Y[train_index],
                     batch_size=cfg.model_train_batch_size(),
                     epochs=cfg.model_train_epoches(),
                     verbose=0, class_weight=cfg.model_class_weight(),
                     validation_data=([X[test_index], X[test_index]], Y[test_index]),
                     callbacks=[early_stopping, checkpointer]
                     )
        scores = model_nn.evaluate([X[test_index], X[test_index]], Y[test_index], verbose=0)
        print("%s: %.2f%%" % (model_nn.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) < 1:
        print("Expect input argument: config file path.")
        sys.exit()

    main(argv)
