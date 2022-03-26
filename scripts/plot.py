import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import List, Tuple
from enum import Enum


class Parameter(Enum):
    loss = 'loss'
    avg_loss = 'avg_loss'
    map = 'map'

    def __str__(self):
        return self.value


def argument_parser():
    parser = argparse.ArgumentParser(description='The script to download the dobble dataaset')
    parser.add_argument('--log-file-path',
                        type=Path,
                        required=True,
                        default='',
                        help='Path to training logs')
    parser.add_argument('--parameter',
                        type=Parameter,
                        required=True,
                        help='Parameter to plot',
                        choices=list(Parameter))
    parser.add_argument('--output',
                        type=Path,
                        default='plot.png',
                        help='Path to result')
    return parser.parse_args()



def read_data(pattern, x_label_group: str, y_label_group:str, log_data: List[str]) -> Tuple[List[float], List[float]]:
    
    x_values = []
    y_values = []

    for line in log_data:
        match = pattern.match(line)
        if not match:
            continue
        x_values.append(float(match.group(x_label_group)))
        y_values.append(float(match.group(y_label_group)))
        
    return x_values, y_values

def get_loss_pattern() -> Tuple[List[int], str, str]:
    x_group_name = 'iteration'
    y_group_name = 'loss'
    return (
        re.compile(rf'\s(?P<{x_group_name}>\d+):\s(?P<{y_group_name}>\d+.\d+),\s\d+.\d+\savg\sloss.*'),
        x_group_name,
        y_group_name
    )

def get_avg_loss_pattern() -> Tuple[List[int], str, str]:
    x_group_name = 'iteration'
    y_group_name = 'avg_loss'
    return (
        re.compile(rf'\s(?P<{x_group_name}>\d+):\s\d+.\d+,\s(?P<{y_group_name}>\d+.\d+)\savg\sloss.*'),
        x_group_name,
        y_group_name
    )


def get_map_pattern() -> Tuple[List[int], str, str]:
    x_group_name = 'iteration'
    y_group_name = 'mAP'
    return (
        re.compile(rf'\siteration\s(?P<{x_group_name}>\d+)\s:mean_average_precision\s\(mAP@0.50\)\s=\s(?P<{y_group_name}>\d\.\d+).*'),
        x_group_name,
        y_group_name
    )

def get_pattern(parameter: Parameter) -> Tuple:
    parameter_per_pattern = {
        Parameter.loss: get_loss_pattern,
        Parameter.map: get_map_pattern
    }
    return parameter_per_pattern[parameter]()

def read_data(log_file_path: Path, parameter: Parameter) -> Tuple[List[float], List[float]]:
    pattern, x_group_name, y_group_name = get_pattern(parameter) 
    with log_file_path.open() as log_file:
        log_data = log_file.readlines()
    
    x_values = []
    y_values = []

    for line in log_data:
        match = pattern.match(line)
        if not match:
            continue
        dir(match)
        x_value = float(match.group(x_group_name))
        y_value = float(match.group(y_group_name))

        x_values.append(x_value)
        y_values.append(y_value)

    return x_values, y_values


def main(log_file_path: Path, parameter: Parameter, output_path: Path):
    x, y = read_data(log_file_path, parameter)

    plt.plot(x, y)
    plt.plot(x, y)
    plt.ylabel('mAP')
    plt.xlabel('Iterations')
    plt.grid()

    plt.savefig(output_path)



if __name__ == '__main__':
    args = argument_parser()
    main(args.log_file_path, args.parameter, args.output)