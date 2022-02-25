import argparse
from pathlib import Path
import zipfile
import shutil

import requests

from common import get_default_dataset_folder

def argument_parser():
    parser = argparse.ArgumentParser(description='The script to download the dobble dataaset')
    parser.add_argument('--dataset-link',
                        type=str,
                        required=True,
                        default='',
                        help='Link to the source dataset')
    parser.add_argument('--result-folder',
                        type=Path,
                        default=get_default_dataset_folder(),
                        help='Full path to the result dataset')
    return parser.parse_args()



def download(dataset_link: Path, result_folder: Path) -> Path:
    if result_folder.exists():
        shutil.rmtree(result_folder)
    
    result_folder.mkdir()

    dataset_result_file = result_folder / 'dataset.zip'
    
    request = requests.get(dataset_link, stream=True)
    with open(dataset_result_file, 'wb') as fd:
        for chunk in request.iter_content(chunk_size=128):
            fd.write(chunk)

    return dataset_result_file


def unzip(dataset_archive_path: Path, result_folder: Path):
    with zipfile.ZipFile(dataset_archive_path, 'r') as zip_ref:
        zip_ref.extractall(result_folder)

def remove_archive(dataset_archive_path: Path):
    if dataset_archive_path.exists:
        dataset_archive_path.unlink()

def main(dataset_link: Path, result_path: Path):
    dataset_archive = download(dataset_link, result_path) 
    unzip(dataset_archive, result_path)
    remove_archive(dataset_archive)


if __name__ == '__main__':
    args = argument_parser()
    main(dataset_link=args.dataset_link, result_path=args.result_folder)

    
