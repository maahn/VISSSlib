import os
import socket

import VISSSlib


def get_test_path():
    path = os.path.dirname(__file__)
    return path


def get_test_data_path():
    path = os.path.join(get_test_path(), "data")
    return path


def readTestSettings(settings):
    testRoot = get_test_data_path()
    settings = f"{testRoot}/{settings}"
    config = VISSSlib.tools.readSettings(settings)
    config["tmpPath"] = f"{os.path.dirname(settings)}/tmp_{socket.gethostname()}"
    config["fileQueue"] = f"{config['tmpPath']}/fileQueue"
    return config
