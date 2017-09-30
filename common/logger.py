import logging


def init_log(severity_level):
    logging.basicConfig(level=severity_level,
                        format='%(asctime)s %(filename)s %(message)s')