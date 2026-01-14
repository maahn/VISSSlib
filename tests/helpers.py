import os
import socket

import VISSSlib


def get_test_data_path():
    path = os.path.join(os.path.dirname(__file__), "data")
    print(path)
    return path


def readTestSettings(settings):
    testRoot = get_test_data_path()
    settings = f"{testRoot}/{settings}"
    config = VISSSlib.tools.readSettings(settings)
    config["tmpPath"] = f"{os.path.dirname(settings)}/tmp_{socket.gethostname()}"
    config["fileQueue"] = f"{config['tmpPath']}/fileQueue"
    return config
