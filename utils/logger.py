import logging
import os

class Logger(object):
    def __init__(self, log_path, log_level=logging.INFO, logger_name="Yolov4"):
        # firstly, create a logger
        self.__log_dir_init(log_path)
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        # secondly, create a handler
        file_handler = logging.FileHandler(log_path)
        # thirdly, define the output form of handler
        formatter = logging.Formatter(
            "[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s "
        )
        file_handler.setFormatter(formatter)
        # finally, add the Hander to logger
        self.__logger.addHandler(file_handler)

    def info_both(self, text):
        """ print text to both console and log file """
        print(text)
        self.__logger.info(text)

    def info(self, text):
        """ print text to log file """
        self.__logger.info(text)

    def __log_dir_init(self, log_path):
        flag = False
        log_dir = os.path.dirname(log_path)
        if os.path.exists(log_dir):
            if os.path.exists(log_path):
                print('The specified log_path already exists.')
                c = input('Continue? (y/[n])? ')
                if c in ['y', 'Y']:
                    flag = True
            else:
                flag = True
        else:
            os.makedirs(log_dir)
            flag = True
        if not flag:
            exit()
        print('Log file: %s' % log_path)

