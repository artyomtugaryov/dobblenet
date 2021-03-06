import argparse
from pathlib import Path
import shutil

from common import get_default_dataset_folder, get_darknet_folder, get_current_folder


def argument_parser():
    parser = argparse.ArgumentParser(description='The script spreads the dataset on the Darknet repository folder')
    parser.add_argument('--dataset-path',
                        type=Path,
                        default=get_default_dataset_folder(),
                        help='Link to the source dataset')
    parser.add_argument('--darknet-folder',
                        type=Path,
                        default=get_darknet_folder(),
                        help='Full path to the result dataset')
    return parser.parse_args()


def copy_obj_folder(dataset_root_path: Path, darknet_root_path: Path):
    obj_folder = dataset_root_path / 'obj'
    darknet_obj_folder = darknet_root_path / 'data' / 'obj'

    if not obj_folder.is_dir():
        raise FileNotFoundError('Direcory {dataset_root_path} is not exist')
    
    if darknet_obj_folder.is_dir():
        shutil.rmtree(darknet_obj_folder)

    shutil.copytree(obj_folder, darknet_obj_folder)


def copy_file(source_file_path: Path, result_file_path: Path):
    shutil.copy(source_file_path, result_file_path)


def adjust_data_file(data_file_path: Path, darknet_data_folder: Path, all=True):
    with data_file_path.open() as obj_data_file:
        obj_data_content = obj_data_file.readlines()
    
    result_content = []
    for line in obj_data_content:
        separate_line = line.split(' ')
        first_element = separate_line[0]
        result = line
        if not line.isspace() and all or first_element in {'train', 'valid', 'test', 'names'}:
            last_element = separate_line[-1]
            new_path = darknet_data_folder / last_element
            result = line.replace(str(last_element), str(new_path))
        result_content.append(result)
    
    with data_file_path.open('w') as obj_data_file:
        obj_data_file.writelines(result_content)


def copy_dataset_metadata(dataset_path: Path, darknet_root_path: Path):
    darknet_data_folder = darknet_root_path / 'data'
    
    source_obj_names_file_path = dataset_path / 'obj.names'
    darknet_obj_names_file_path = darknet_data_folder / 'obj.names'
    copy_file(source_obj_names_file_path, darknet_obj_names_file_path)

    source_obj_data_file_path = dataset_path / 'obj.data'
    darknet_obj_data_file_path = darknet_data_folder / 'obj.data'
    copy_file(source_obj_data_file_path, darknet_obj_data_file_path)
    adjust_data_file(darknet_obj_data_file_path, darknet_data_folder, all=False)

    source_test_file_path = dataset_path / 'test.txt'
    darknet_test_file_path = darknet_data_folder / 'test.txt'
    copy_file(source_test_file_path, darknet_test_file_path)
    adjust_data_file(darknet_test_file_path, darknet_data_folder)

    source_valid_file_path = dataset_path / 'valid.txt'
    darknet_valid_file_path = darknet_data_folder / 'valid.txt'
    copy_file(source_valid_file_path, darknet_valid_file_path)
    adjust_data_file(darknet_valid_file_path, darknet_data_folder)


    source_train_file_path = dataset_path / 'train.txt'
    darknet_train_file_path = darknet_data_folder / 'train.txt'
    copy_file(source_train_file_path, darknet_train_file_path)
    adjust_data_file(darknet_train_file_path, darknet_data_folder)

    
def main(dataset_path: Path, darknet_root_path: Path):
    copy_obj_folder(dataset_path, darknet_root_path)
    copy_dataset_metadata(dataset_path, darknet_root_path)


if __name__ == '__main__':
    args = argument_parser()
    main(args.dataset_path, args.darknet_folder)

