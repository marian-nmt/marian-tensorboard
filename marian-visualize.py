#!/usr/bin/env python3

import argparse
import datetime
import pickle
import sys
import tensorboardX as tb
import time


class MarianLogReader(object):
    def __init__(self):
        pass

    def parse_file(self, path):
        with open(path, "r", encoding="utf-8") as logs:
            for line in logs:
                output = self.parse_line(line)
                if output:
                    sys.stderr.write(f"{output}\n")
                    yield output

    def parse_line(self, line):
        if "[valid]" in line:
            (
                _,
                _,
                _,
                _ep,
                ep,
                _,
                _up,
                up,
                _,
                metric,
                _,
                value,
                _,
                _stalled,
                x,
                *_times,
            ) = line.split()
            update = int(up)
            value = float(value)
            return (datetime.datetime.now(), update, metric, value)
        return None


class LogWriter(object):
    def __init__(self):
        pass


class TensorboardWriter(LogWriter):
    def __init__(self, path):
        self.writer = tb.SummaryWriter(path)

    def write(self, timestamp, update, metric, value):
        self.writer.add_scalar(f"valid/{metric}", value, update, 123)

    def close(self):
        self.writer.close()


class AzureMLMetricsWriter(LogWriter):
    pass


def main():
    args = parse_user_args()

    reader = MarianLogReader()
    writer = TensorboardWriter(args.log_dir)
    for tup in reader.parse_file(args.log_file):
        writer.write(*tup)


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", help="path to train.log", default="train.log")
    parser.add_argument(
        "--log-dir", help="logging directory for tensorboard", default="logdir"
    )
    parser.add_argument("--port", help="", default="logdir")
    parser.add_argument(
        "--update-freq", help="update frequency in seconds", type=int, default=5
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
