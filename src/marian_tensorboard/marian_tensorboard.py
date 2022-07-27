#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import calendar
import logging
import os
import pickle
import re
import signal
import sys
import tensorboard as tb
import tensorboardX as tbx
import threading
import time

from functools import reduce
from pathlib import Path
from version import __version__ as VERSION

# Setup logger suppressing logging from external modules
logger = logging.getLogger("marian-tensorboard")
logging.basicConfig(level=logging.ERROR)


class LogFileReader(object):
    """Reader for log files in a text format."""

    def __init__(self, path, workdir):
        self.log_file = Path(path)
        if workdir:
            self.state_file = Path(workdir) / "state.pkl"
        else:
            self.state_file = None
        self.last_update = 0
        self.last_line = 0

        self._load_state()

        logger.info(
            f"Log file {self.log_file} "
            + f"last updated at {self.last_update}, "
            + f"previously processed lines: {self.last_line}"
        )

    def read(self):
        """Reads new lines added since the last read."""
        if not self._need_update():
            logger.debug(f"No need to update {self.log_file}")
            return
        with open(self.log_file, "r", encoding="utf-8") as logs:
            for line_no, line in enumerate(logs):
                if self.last_line and self.last_line < line_no:
                    yield line
                if line_no > self.last_line:
                    self.last_line = line_no
                if self.log_file.stat().st_mtime > self.last_update:
                    self.last_update = self.log_file.stat().st_mtime
            self._save_state()

    def _load_state(self):
        if Path(self.state_file).exists():
            with open(self.state_file, "rb") as fstate:
                self.last_update, self.last_line = pickle.load(fstate)

    def _save_state(self):
        with open(self.state_file, "wb") as fstate:
            pickle.dump((self.last_update, self.last_line), fstate)

    def _need_update(self):
        # logger.debug(f"Last update: {self.last_update}, last touch: {self.log_file.stat().st_mtime}")
        if self.last_update > 0 and self.last_update >= self.log_file.stat().st_mtime:
            return False
        return True


class MarianLogParser(object):
    """Parser for Marian logs."""

    def __init__(self):
        self.total_sentences = 0
        self.train_re = re.compile(
            r"Ep\.[\s]+(?P<epoch>[\d]+)[\s]+:[\s]Up\.[\s](?P<updates>[\d]+)[\s]+:[\s]Sen\.[\s](?P<sentences>[0-9|,]+).*?(?P<metric>[A-z|-]+)[\s]+(?P<value>[\d\.]+)"
        )
        self.valid_re = re.compile(
            r"\[valid\][\s]+Ep\.[\s]+(?P<epoch>[\d]+)[\s]+:[\s]Up\.[\s](?P<updates>[\d]+).*?(?P<metric>[a-z|-]+)[\s]+:[\s]+(?P<value>[\d\.]+)([\s]+:[\s]stalled[\s](?P<stalled>[\d]+))?"
        )
        self.config_re = re.compile(
            r"\[config\].*?(?P<config_name>[A-z|-]+):[\s]+(?P<config_value>[\d\.|A-z]+)"
        )
        self.total_sentences_re = re.compile(r"Seen[\s]+(?P<epoch_sentence>[\d]+)")

    def parse_line(self, line):
        """
        Parses a log line and returns tuple(s) of (time, update, metric, value).
        """
        m = self.config_re.search(line)
        if m:
            config_name = m.group("config_name")
            config_value = m.group("config_value")
            yield ("text", None, None, config_name, config_value)

        m = self.valid_re.search(line)
        if m:
            _date, _time, *rest = line.split()
            epoch = int(m.group("epoch"))
            update = int(m.group("updates"))
            metric = m.group("metric")
            value = float(m.group("value"))
            stalled = int(m.group("stalled") or 0)
            yield (
                "scalar",
                self.wall_time(_date + " " + _time),
                update,
                f"valid/{metric}",
                value,
            )
            yield (
                "scalar",
                self.wall_time(_date + " " + _time),
                update,
                f"valid/{metric}_stalled",
                stalled,
            )

        m = self.train_re.search(line)
        if m:
            _date, _time, *rest = line.split()
            epoch = int(m.group("epoch"))
            update = int(m.group("updates"))
            sentences = int(str(m.group("sentences")).replace(",", ""))
            metric = m.group("metric")
            value = float(m.group("value"))
            yield (
                "scalar",
                self.wall_time(_date + " " + _time),
                update,
                "train/epoch",
                epoch,
            )
            yield (
                "scalar",
                self.wall_time(_date + " " + _time),
                update,
                f"train/{metric}",
                value,
            )
            yield (
                "scalar",
                self.wall_time(_date + " " + _time),
                update,
                f"train/update_sent",
                sentences,
            )
            yield (
                "scalar",
                self.wall_time(_date + " " + _time),
                update,
                f"train/total_sent",
                sentences + self.total_sentences,
            )

        m = self.total_sentences_re.search(line)
        if m:
            epoch_sentence = int(m.group("epoch_sentence"))
            self.total_sentences += epoch_sentence

    def wall_time(self, string):
        return calendar.timegm(time.strptime(string, "[%Y-%m-%d %H:%M:%S]"))


class LogWriter(object):
    """Template class for logging writers."""

    def write(self, type, time, update, metric, value):
        raise NotImplemented


class TensorboardWriter(LogWriter):
    """Writing logs for TensorBoard using TensorboardX."""

    def __init__(self, path):
        self.writer = tbx.SummaryWriter(path)
        logger.info(f"Exporting to Tensorboard directory: {path}")

    def write(self, type, time, update, metric, value):
        if type == "scalar":
            self.writer.add_scalar(metric, value, update, time)
        elif type == "text":
            self.writer.add_text(metric, value)
        else:
            raise NotImplemented


class AzureMLMetricsWriter(LogWriter):
    """Writing logs for Azure ML metrics."""

    def __init__(self):
        from azureml.core import Run

        self.writer = Run.get_context()
        logger.info("Logging to AzureML Metrics...")

    def write(self, type, time, update, metric, value):
        if type == "scalar":
            self.writer.log_row(metric, x=update, y=value)
        else:
            pass


class ConvertionJob(threading.Thread):
    """Job connecting logging readers and writers in a subthread."""

    def __init__(self, log_file, work_dir, update_freq=5, azureml=False):
        threading.Thread.__init__(self)

        # The shutdown_flag is a threading.Event object that
        # indicates whether the thread should be terminated.
        self.shutdown_flag = threading.Event()

        self.log_file = log_file
        self.work_dir = work_dir
        self.update_freq = update_freq
        self.azureml = azureml

    def run(self):
        """Runs the convertion job."""
        logging.debug(f"Thread #{self.ident} handling {self.log_file} started")

        log_dir = Path(self.work_dir) / self._abs_path_to_dir_name(self.log_file)
        reader = LogFileReader(path=self.log_file, workdir=log_dir)
        parser = MarianLogParser()

        writers = []
        writers.append(TensorboardWriter(log_dir))
        if self.azureml:
            writers.append(AzureMLMetricsWriter())

        while not self.shutdown_flag.is_set():
            for line_no, line in enumerate(reader.read()):
                for log_tuple in parser.parse_line(line):
                    logger.debug(f"{self.log_file}:{line_no} produced {log_tuple}")
                    for writer in writers:
                        writer.write(*log_tuple)

            if self.update_freq == 0:  # just a single iteration if requested
                break

            time.sleep(self.update_freq)

        logging.debug(f"Thread #{self.ident} stopped")

    def _abs_path_to_dir_name(self, path):
        normalizations = {"/": "__", "\\": "__", " ": ""}
        tmp_path = str(Path(path).resolve().with_suffix(""))
        nrm_path = reduce(
            lambda x, y: x.replace(y[0], y[1]), normalizations.items(), tmp_path
        )
        logger.debug(f"Normalized '{path}' to '{nrm_path}'")
        return nrm_path


class ServiceExit(Exception):
    """Custom exception for signal handling."""

    pass


def main():
    args = parse_user_args()

    # Setup signal handling
    def service_shutdown(signum, frame):
        raise ServiceExit

    signal.signal(signal.SIGTERM, service_shutdown)
    signal.signal(signal.SIGINT, service_shutdown)

    try:
        # Create a convertion job for each log file
        jobs = []
        for log_file in args.log_file:
            update_freq = 0 if args.offline else 5
            job = ConvertionJob(log_file, args.work_dir, update_freq, args.azureml)
            job.start()
            jobs.append(job)

        if args.offline:
            for job in jobs:
                job.join()
            logger.info("Done")

        if not args.azureml:
            logger.info("Starting TensorBoard server...")
            launch_tensorboard(args.work_dir, args.port)  # Start teansorboard

        while True:  # Keep the main thread running so that signals are not ignored
            time.sleep(0.5)

    except ServiceExit:
        logger.info("Exiting... it may take a few seconds")
        for job in jobs:
            job.shutdown_flag.set()
        for job in jobs:
            job.join()

    logger.info("Done")


def launch_tensorboard(logdir, port):
    """Launches TensorBoard server."""
    tb_server = tb.program.TensorBoard()
    tb_server.configure(argv=[None, "--logdir", logdir, "--port", str(port)])
    tb_server.launch()


def parse_user_args():
    """Defines and parses user command line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--log-file",
        nargs="+",
        help="path to train.log files/directory",
        required=True,
    )
    parser.add_argument(
        "-w",
        "--work-dir",
        help="TensorBoard logging directory, default: %(default)s",
        default="logdir",
    )
    parser.add_argument(
        "-p",
        "--port",
        help="port number for TensorBoard, default: %(default)s",
        type=int,
        default=6006,
    )
    parser.add_argument(
        "--offline", help="do not monitor for log updates", action="store_true"
    )
    parser.add_argument(
        "--azureml",
        help="generate Azure ML Metrics; updates --work-dir automatically",
        action="store_true",
    )
    parser.add_argument("--debug", help="print debug messages", action="store_true")
    parser.add_argument(
        "--version", action='version', version='%(prog)s {}'.format(VERSION)
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("marian-tensorboard").setLevel(logging.DEBUG)
    else:
        logging.getLogger("marian-tensorboard").setLevel(logging.INFO)

    if args.azureml:
        args.work_dir = os.getenv('AZUREML_TB_PATH')
        logger.info("AzureML RunID: %s" % os.getenv("AZUREML_RUN_ID"))
        logger.info("AzureML Setting TensorBoard logdir: %s" % args.work_dir)

    return args


if __name__ == "__main__":
    main()
