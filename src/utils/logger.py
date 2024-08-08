#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import logging.config
from .file import check_dir


class Logger(object):

    def __init__(self, log_dir_path, logger_name=None, log_file_name=None):
        self.log_dir_path = log_dir_path
        check_dir(self.log_dir_path)
        self.logger_name = logger_name
        self.log_file_name = log_file_name
        self.formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")

        self.init_logger()
        self.add_console_handler()
        self.add_file_handler()

    def init_logger(self):
        logger_name = "logger" if self.logger_name is None else self.logger_name
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

    def add_console_handler(self):

        handler_console = logging.StreamHandler()
        handler_console.setLevel(logging.INFO)
        handler_console.setFormatter(self.formatter)
        self.logger.addHandler(handler_console)

    def add_file_handler(self):
        now = datetime.datetime.now()
        datatime_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        log_file_name = f"log_{datatime_string}.txt" if self.log_file_name is None else self.log_file_name
        log_file_path = os.path.join(self.log_dir_path, log_file_name)
        handler_file = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            mode="a",
            maxBytes=10485760,
            backupCount=100,
            encoding="utf-8"
        )
        handler_file.setLevel(logging.INFO)
        handler_file.setFormatter(self.formatter)
        self.logger.addHandler(handler_file)

    def __call__(self, message, level="info"):
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)
        else:
            assert False
