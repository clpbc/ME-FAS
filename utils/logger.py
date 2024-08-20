# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode = None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal = True, is_file = False):
        if '\r' in message:
            is_file = False
        if is_terminal == True:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == True:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass