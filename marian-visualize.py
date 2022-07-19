#!/usr/bin/env python3

import argparse
import calendar
import datetime
import sys
import tensorboard as tb
import tensorboardX as tbx
import threading
import time

from pathlib import Path


class LogFileReader(object):
    def __init__(self, parser):
        self.parser = parser

    def read(self, path):
        with open(path, "r", encoding="utf-8") as logs:
            for line in logs:
                for output in self.parser.parse_line(line):
                    sys.stderr.write(f"{output}\n")
                    yield output


class MarianLogParser(object):
    """Parser for Marian logs."""

    def parse_line(self, line):
        """
        Parses a log line and returns tuple(s) of (time, update, metric, value).
        """
        if "[valid]" in line:
            (
                _date,
                _time,
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
            yield (self.wall_time(_date + " " + _time), update, metric, value)

    def wall_time(self, string):
        return calendar.timegm(time.strptime(string, "[%Y-%m-%d %H:%M:%S]"))


class LogWriter(object):
    def write(self, time, update, metric, value):
        raise NotImplemented


class TensorboardWriter(LogWriter):
    def __init__(self, path):
        self.writer = tbx.SummaryWriter(path)

    def write(self, time, update, metric, value):
        self.writer.add_scalar(f"valid/{metric}", value, update, time)


class AzureMLMetricsWriter(LogWriter):
    pass


def main():
    args = parse_user_args()

    reader = LogFileReader(parser=MarianLogParser())
    for log_file in args.log_file:
        log_dir = Path(args.tb_logdir) / Path(log_file).stem
        writer = TensorboardWriter(log_dir)
        for tup in reader.read(log_file):
            writer.write(*tup)

    if args.offline:
        print("Done")
        exit()

    print(f"Starting TensorBoard...")
    launch_tensorboard(args)
    # thread = threading.Thread(target=launch_tensorboard, args=([args]))
    # thread.start()

    # Do other stuff...

    # thread.join()
    print("Done")


def launch_tensorboard(args):
    tb_server = tb.program.TensorBoard()
    tb_server.configure(
        argv=[None, '--logdir', args.tb_logdir, '--port', str(args.tb_port)]
    )
    tb_server.main()


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", nargs="+", help="path to train.log files")
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
