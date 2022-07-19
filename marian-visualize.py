#!/usr/bin/env python3

import argparse
import calendar
import pickle
import signal
import sys
import tensorboard as tb
import tensorboardX as tbx
import threading
import time

from pathlib import Path


class LogFileReader(object):
    def __init__(self, path, parser, workdir):
        self.log_file = Path(path)
        self.parser = parser
        if workdir:
            self.state_file = Path(workdir) / "state.pkl"
        else:
            self.state_file = None
        self.last_update = None
        self.last_line = None

        self._load_state()

        sys.stderr.write(
            f"Log file {self.log_file} last updated at {self.last_update}\n"
        )
        sys.stderr.write(f"Previously processed lines: {self.last_line}\n")

    def read(self):
        if not self._need_update():
            sys.stderr.write(f"No need to update {self.log_file}...\n")
            return
        with open(self.log_file, "r", encoding="utf-8") as logs:
            for line_no, line in enumerate(logs):
                if self.last_line and self.last_line < line_no:
                    for output in self.parser.parse_line(line):
                        sys.stderr.write(f"{output}\n")
                        yield output
                self.last_line = line_no
                self.last_update = self.log_file.stat().st_mtime
            self._save_state()

    def _load_state(self):
        if not Path(self.state_file).exists():
            return
        with open(self.state_file, "rb") as fstate:
            self.last_update, self.last_line = pickle.load(fstate)

    def _save_state(self):
        with open(self.state_file, "wb") as fstate:
            pickle.dump((self.last_update, self.last_line), fstate)

    def _need_update(self):
        if self.last_update and self.last_update <= self.log_file.stat().st_mtime:
            return False
        return True


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


class ConvertionJob(threading.Thread):
    def __init__(self, log_file, work_dir, update_freq=5):
        threading.Thread.__init__(self)

        # The shutdown_flag is a threading.Event object that
        # indicates whether the thread should be terminated.
        self.shutdown_flag = threading.Event()

        self.log_file = log_file
        self.work_dir = work_dir
        self.update_freq = update_freq

    def run(self):
        print(f"Thread #{self.ident} handling {self.log_file} started")

        log_dir = Path(self.work_dir) / Path(self.log_file).stem
        reader = LogFileReader(
            path=self.log_file, workdir=log_dir, parser=MarianLogParser()
        )
        writer = TensorboardWriter(log_dir)

        for tup in reader.read():
            writer.write(*tup)

        while self.update_freq > 0 and not self.shutdown_flag.is_set():
            for tup in reader.read():
                writer.write(*tup)
            time.sleep(self.update_freq)

        print(f"Thread #{self.ident} stopped")


class ServiceExit(Exception):
    pass


def main():
    args = parse_user_args()

    def service_shutdown(signum, frame):
        raise ServiceExit

    signal.signal(signal.SIGTERM, service_shutdown)
    signal.signal(signal.SIGINT, service_shutdown)

    try:
        jobs = []
        for log_file in args.log_file:
            update_freq = 0 if args.offline else 5
            job = ConvertionJob(log_file, args.tb_logdir, update_freq)
            job.start()
            jobs.append(job)

        if args.offline:
            for job in jobs:
                job.join()
            print("Done")

        print(f"Starting TensorBoard...")
        launch_tensorboard(args)

        while True:  # Keep the main thread running so that signals are not ignored
            time.sleep(0.5)

    except ServiceExit:
        print("Exiting... it may take a few seconds")
        for job in jobs:
            job.shutdown_flag.set()
        for job in jobs:
            job.join()

    print("Done")


def setup_convertion(log_file, work_dir):
    log_dir = Path(work_dir) / Path(log_file).stem
    reader = LogFileReader(path=log_file, workdir=log_dir, parser=MarianLogParser())
    writer = TensorboardWriter(log_dir)
    while True:
        for tup in reader.read():
            writer.write(*tup)
        time.sleep(5)


def launch_tensorboard(args):
    tb_server = tb.program.TensorBoard()
    tb_server.configure(
        argv=[None, '--logdir', args.tb_logdir, '--port', str(args.tb_port)]
    )
    # tb_server.main()
    tb_server.launch()


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
