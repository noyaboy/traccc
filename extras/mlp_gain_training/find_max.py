#!/usr/bin/env python3
"""
Script to parse a result.log file and find the minimal FP32 and INT8 test losses
along with their corresponding hyperparameters.
"""
import re
import argparse


def parse_log(file_path):
    # Regular expressions for parsing
    pattern_run = re.compile(
        r"Running batch_size=(?P<batch_size>\d+)\s+lr=(?P<lr>[\d\.eE+-]+)\s+weight_decay=(?P<wd>[\d\.eE+-]+)"
    )
    pattern_fp32 = re.compile(r"FP32 test loss:(?P<loss>[\d\.eE+-]+)")
    pattern_int8 = re.compile(r"INT8 test loss:(?P<loss>[\d\.eE+-]+)")

    # Initialize minima
    min_fp32 = float('inf')
    min_fp32_params = None
    min_int8 = float('inf')
    min_int8_params = None

    current_params = None

    with open(file_path, 'r') as f:
        for line in f:
            # Check for a new hyperparameter setting
            m_run = pattern_run.search(line)
            if m_run:
                current_params = m_run.groupdict()
                continue

            # Check for FP32 loss under current setting
            m_fp32 = pattern_fp32.search(line)
            if m_fp32 and current_params is not None:
                loss = float(m_fp32.group('loss'))
                if loss < min_fp32:
                    min_fp32 = loss
                    min_fp32_params = current_params.copy()
                continue

            # Check for INT8 loss under current setting
            m_int8 = pattern_int8.search(line)
            if m_int8 and current_params is not None:
                loss = float(m_int8.group('loss'))
                if loss < min_int8:
                    min_int8 = loss
                    min_int8_params = current_params.copy()
                continue

    return min_fp32, min_fp32_params, min_int8, min_int8_params


def run():
    """
    Parse command-line arguments, invoke parse_log, and print the minimal losses.
    """
    parser = argparse.ArgumentParser(
        description='Find minimal FP32 and INT8 test losses in a log file.'
    )
    parser.add_argument(
        'log_file', type=str, help='Path to the result.log file to parse'
    )
    args = parser.parse_args()

    min_fp32, fp32_params, min_int8, int8_params = parse_log(args.log_file)

    if fp32_params:
        print(f"Minimal FP32 test loss: {min_fp32}")
        print(
            f"Parameters: batch_size={fp32_params['batch_size']}, "
            f"lr={fp32_params['lr']}, weight_decay={fp32_params['wd']}"
        )
    else:
        print("No FP32 test loss entries found.")

    print()  # Blank line between results

    if int8_params:
        print(f"Minimal INT8 test loss: {min_int8}")
        print(
            f"Parameters: batch_size={int8_params['batch_size']}, "
            f"lr={int8_params['lr']}, weight_decay={int8_params['wd']}"
        )
    else:
        print("No INT8 test loss entries found.")


def main():
    # 在 main 中只呼叫一個 run() 函數
    run()


if __name__ == '__main__':
    main()
