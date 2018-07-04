import os

class LogFileAlreadyExists (Exception):
    def __init__(self, path):
        super(LogFileAlreadyExists, self).__init__('The log file {} already exists'.format(path))


class Logger (object):
    def __init__(self, log_file, default_path):
        """
        Simple logger

        Prints output as well as writing it to a file.

        Raises an exception if the log file already exists

        Pass the log file path received from the command line into the log_file argument.
        If it is empty, `default_path` will be used.
        If it is 'none' then log messages will be printed to stdout but not saved

        :param log_file: Log file path;  'none' for no logging, empty ('') to use `default_path`
        :param default_path: default path to use if `log_file` is empty
        """
        if log_file == '':
            log_file = default_path
        elif log_file == 'none':
            log_file = None

        if log_file is not None:
            if os.path.exists(log_file):
                raise LogFileAlreadyExists(log_file)

        self.log_file = log_file
        self._dir_checked = False

    def log(self, text):
        print(text)
        if self.log_file is not None:
            if not self._dir_checked:
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                self._dir_checked = True

            with open(self.log_file, 'a') as f:
                f.write(text + '\n')
                f.flush()
                f.close()

    def __call__(self, text):
        return self.log(text)