import logging


def set_logger(log_filename):
    if len(logging.getLogger().handlers) == 0:
        log_format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
        # 打印到控制台
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)  # 设置控制台日志输出的级别。如果设置为logging.INFO，就不会输出DEBUG日志信息
        console.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(console)

    if len(logging.getLogger().handlers) > 1:
        logger = logging.getLogger()
        logger.handlers[1].stream.close()
        logger.removeHandler(logger.handlers[1])

    log_format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    logging.getLogger().setLevel(logging.DEBUG)

    # 自动换文件
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
