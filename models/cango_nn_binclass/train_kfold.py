import sys
import os
import keras as ka
import numpy as np
from common import logger, constants, config
from datasets import cango_pboc
from models.cango_nn_binclass import model
from sklearn.model_selection import StratifiedKFold


config_file = '../../configs/cango_nn_raw.yaml'


def main(argv):
    cfg = config.YamlParser(config_file)
    log_dir, out_dir = logger.init(log_dir=cfg.log_dir(),
                                   out_dir=cfg.out_dir(),
                                   level=cfg.log_level())
    weight_path = '{}/weights.h5'.format(out_dir)

    (X, Y), (_, _) = cango_pboc.get_train_val_data(
        path=cfg.train_data(), drop_columns=cfg.drop_columns(),
        train_val_ratio=0.0,
        do_shuffle=cfg.do_shuffle(), do_smote=cfg.do_smote(), smote_ratio=cfg.smote_ratio())

    kfold = StratifiedKFold(n_splits=10, shuffle=True,
                            random_state=constants.random_seed)

    checkpointer = ka.callbacks.ModelCheckpoint(filepath=weight_path,
                                                verbose=1,
                                                save_best_only=True)
    early_stopping = ka.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0, patience=5,
                                                verbose=1, mode='auto')

    cvscores = []
    for train_index, test_index in kfold.split(X, Y):
        # Construct the model
        input_dim = X.shape[1]
        model_nn = model.create_model(input_dim,
                                      regularization_val=cfg.model_reg_val(),
                                      dropout_val=cfg.model_dropout_val(),
                                      learning_rate=cfg.model_learning_rate())

        if os.path.exists(weight_path):
            model_nn.load_weights(weight_path)

        model_nn.fit(X[train_index], Y[train_index],
                     batch_size=cfg.model_train_batch_size(),
                     epochs=cfg.model_train_epoches(),
                     verbose=0,
                     validation_data=(X[test_index], Y[test_index]),
                     callbacks=[early_stopping, checkpointer]
                     )
        scores = model_nn.evaluate(X[test_index], Y[test_index], verbose=0)
        print("%s: %.2f%%" % (model_nn.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


if __name__ == '__main__':
    main(sys.argv[1:])
