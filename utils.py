import os


def get_name(path):
    return os.path.splitext(os.path.basename(path))[0]
