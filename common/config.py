import numpy as np
import yaml
import common.constants as c

class YamlParser:
    def __init__(self, file):
        with open(file, "r") as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def train_data(self):
        return self.config[c.CONFIG_SECT_INPUT]['train_dataset']

    def test_data(self):
        return self.config[c.CONFIG_SECT_INPUT]['test_dataset']

    def train_val_ratio(self):
        return float(self.config[c.CONFIG_SECT_INPUT]['train_val_ratio'])

    def do_shuffle(self):
        return bool(self.config[c.CONFIG_SECT_INPUT]['do_shuffle'])

    def do_smote(self):
        return bool(self.config[c.CONFIG_SECT_INPUT]['do_smote'])

    def drop_columns(self):
        return np.asarray(self.config[c.CONFIG_SECT_INPUT]['drop_columns'])

    def smote_ratio(self):
        return float(self.config[c.CONFIG_SECT_INPUT]['smote_ratio'])

    def log_dir(self):
        return self.config[c.CONFIG_SECT_LOG]['log_dir']

    def log_level(self):
        return self.config[c.CONFIG_SECT_LOG]['level']

    def out_dir(self):
        return self.config[c.CONFIG_SECT_OUTPUT]['out_dir']

    def model_reg_val(self):
        return float(self.config[c.CONFIG_SECT_MODEL]['regularization_val'])

    def model_dropout_val(self):
        return float(self.config[c.CONFIG_SECT_MODEL]['dropout_val'])

    def model_learning_rate(self):
        return float(self.config[c.CONFIG_SECT_MODEL]['learning_rate'])

    def model_train_batch_size(self):
        return int(self.config[c.CONFIG_SECT_MODEL]['train']['batch_size'])

    def model_train_epoches(self):
        return int(self.config[c.CONFIG_SECT_MODEL]['train']['epoches'])

