import logging
import datetime, os


def init(log_dir, out_dir, level='DEBUG'):
    dtstr = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_dir = "{}_{}".format(log_dir, dtstr)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    __init_log(severity_level=level)

    return log_dir, out_dir


def __init_log(severity_level):
    logging.basicConfig(level=severity_level,
                        format='%(asctime)s %(filename)s %(message)s')