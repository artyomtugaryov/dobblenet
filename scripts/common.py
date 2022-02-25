from pathlib import Path


def get_current_folder() -> Path:
    return Path.cwd()


def get_default_dataset_folder() -> Path:
    return get_current_folder() / 'dataset'

def get_darknet_folder() -> Path:
    return get_current_folder() / 'darknet'

def get_darknet_data_folder() -> Path:
    return get_darknet_folder() / 'data'