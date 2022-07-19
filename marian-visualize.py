#!/usr/bin/env python3

import argparse
import datetime
import pickle
import sys
import tensorboard as tb
import tensorboardX as tbx
import time


class LogFileReader(object):
    def __init__(self, parser):
        self.parser = parser

    def read(self, path):
        with open(path, "r", encoding="utf-8") as logs:
            for line in logs:
                output = self.parser.parse_line(line)
                if output:
                    sys.stderr.write(f"{output}\n")
                    yield output


class MarianLogParser(object):
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
    def write(self, time, update, metric, value):
        raise NotImplemented


class TensorboardWriter(LogWriter):
    def __init__(self, path):
        self.writer = tbx.SummaryWriter(path)

    def write(self, time, update, metric, value):
        self.writer.add_scalar(f"valid/{metric}", value, update, 123)


class AzureMLMetricsWriter(LogWriter):
    pass


def main():
    args = parse_user_args()

    reader = LogFileReader(parser=MarianLogParser())
    writer = TensorboardWriter(args.tb_logdir)
    for tup in reader.read(args.log_file):
        writer.write(*tup)

    if args.offline:
        print("Done")
    else:
        print(f"Starting TensorBoard ...")
        tb_server = tb.program.TensorBoard()
        tb_server.configure(
            argv=[None, '--logdir', args.tb_logdir, '--port', str(args.tb_port)]
        )
        tb_server.main()


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", help="path to train.log", default="train.log")
    parser.add_argument(
        "--tb-logdir", help="TensorBoard logging directory", default="logdir"
    )
    parser.add_argument(
        "--tb-port", help="port number for tensorboard", type=int, default=6006
    )
    parser.add_argument("--offline", help="visualize", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
