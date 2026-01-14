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
    if not os.path.exists(settings):
        downloadData()
    config = VISSSlib.tools.readSettings(settings)
    config["tmpPath"] = f"{os.path.dirname(settings)}/tmp_{socket.gethostname()}"
    config["fileQueue"] = f"{config['tmpPath']}/fileQueue"

    return config


def downloadData():
    test_path = get_test_path()
    url = "https://speicherwolke.uni-leipzig.de/public.php/dav/files/PJ8dt77ND9tmaB2/?accept=zip"
    zip_path = os.path.join(test_path, "data.zip")
    print(f"Downloading test data from {url} to {zip_path}")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Extracting test data to {test_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(test_path)
    print(f"Removing zip file {zip_path}")
    os.remove(zip_path)
    return
