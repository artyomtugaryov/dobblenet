import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import List


def argument_parser():
    parser = argparse.ArgumentParser(description='The script to download the dobble dataaset')
    parser.add_argument('--log-file-path',
                        type=Path,
                        required=True,
                        default='',
                        help='Path to training logs')

    parser.add_argument('--output',
                        type=Path,
                        default='plot.png',
                        help='Path to result')
    return parser.parse_args()



def read_data(log_file_path: Path) -> List[str]:

    data_regexp = re.compile('\s(?P<iteration>\d+):\s(?P<loss>\d+.\d+),\s(?P<avg_loss>\d+.\d+)\savg\sloss.*')
    
    with log_file_path.open() as log_file_path:
        lines = log_file_path.readlines()
    iterations = []
    losses = []
    avg_losses = []

    for line in lines:
        match = data_regexp.match(line)
        if not match:
            continue
        iterations.append(int(match.group('iteration')))
        losses.append(float(match.group('loss')))
        avg_losses.append(float(match.group('avg_loss')))

    return iterations, losses, avg_losses


def main(log_file_path: Path, output_path: Path):
    iterations, losses, avg_loss = read_data(log_file_path)

    plt.plot(iterations, losses)
    plt.plot(iterations, avg_loss)
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.grid()

    plt.savefig(output_path)



if __name__ == '__main__':
    args = argument_parser()
    main(args.log_file_path, args.output)