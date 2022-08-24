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

try:
    from .version import __version__ as VERSION
except ImportError:
    VERSION = 'unknown'

from functools import reduce
from pathlib import Path

UPDATE_FREQ = 10  # Monitoring for updates in log files every this number of seconds

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
            r"Ep\.[\s]+(?P<epoch>[\d.]+)[\s]+:[\s]"  # Ep. 1.234 :
            r"Up\.[\s](?P<updates>[\d]+)[\s]+:[\s]"  # Up. 1234 :
            r"Sen\.[\s](?P<sentences>[0-9|,]+).*?"  # Sen. 1,234,567 :
            r"(?P<metric>[A-z|-]+)[\s]+(?P<value>[\d\.]+).*?"  # Cost 1.23456 :
            r"L\.r\.[\s](?P<learnrate>[\d\.]+e-[\d]+)"  # L.r. 1.234-05
        )
        self.valid_re = re.compile(
            r"\[valid\][\s]+"
            r"Ep\.[\s]+(?P<epoch>[\d.]+)[\s]+:[\s]"
            r"Up\.[\s](?P<updates>[\d]+).*?"
            r"(?P<metric>[a-z|-]+)[\s]+:[\s]+(?P<value>[\d\.]+)([\s]+:[\s]stalled[\s](?P<stalled>[\d]+))?"
        )
        self.config_re = re.compile(
            r"\[config\].*?(?P<config_name>[A-z|-]+):[\s]+(?P<config_value>[\d\.|A-z]+)"
        )
        self.total_sentences_re = re.compile(r"Seen[\s]+(?P<epoch_sentence>[\d.]+)")

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
            epoch = float(m.group("epoch"))
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
            epoch = float(m.group("epoch"))
            update = int(m.group("updates"))
            sentences = int(str(m.group("sentences")).replace(",", ""))
            metric = m.group("metric")
            value = float(m.group("value"))
            learnrate = float(m.group("learnrate"))
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
            yield (
                "scalar",
                self.wall_time(_date + " " + _time),
                update,
                f"train/learn_rate",
                learnrate,
            )

        m = self.total_sentences_re.search(line)
        if m:
            epoch_sentence = int(m.group("epoch_sentence"))
            self.total_sentences += epoch_sentence

    def wall_time(self, string):
        """Converts timestamp string into strptime. Strips brackets if necessary."""
        if string.startswith("["):
            string = string[1:]
        if string.endswith("]"):
            string = string[:-1]
        return calendar.timegm(time.strptime(string, "%Y-%m-%d %H:%M:%S"))


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
        logger.info("Logging to Azure ML Metrics...")

    def write(self, type, time, update, metric, value):
        if type == "scalar":
            self.writer.log_row(metric, x=update, y=value)
        else:
            pass


class MLFlowTrackingWriter(LogWriter):
    """Writing logs for MLflow Tracking."""

    def __init__(self):
        import mlflow

        logger.info("Autologging to MLflow...")
        mlflow.autolog()

    def write(self, type, time, update, metric, value):
        if type == "scalar":
            mlflow.log_metric(metric, value, step=update)
        elif type == "text":
            mlflow.log_param(metric, value)
        else:
            pass


class ConvertionJob(threading.Thread):
    """Job connecting logging readers and writers in a subthread."""

    def __init__(self, log_file, work_dir, update_freq=5, azureml=False):
        threading.Thread.__init__(self)

        # The shutdown_flag is a threading.Event object that
        # indicates whether the thread should be terminated.
        self.shutdown_flag = threading.Event()

        self.log_file = Path(log_file)
        self.work_dir = Path(work_dir)
        self.update_freq = update_freq
        self.azureml = azureml

    def run(self):
        """Runs the convertion job."""
        logger = logging.getLogger("marian-tensorboard")
        logger.debug(f"Thread #{self.ident} handling {self.log_file} started")

        log_dir = self.work_dir / self._abs_path_to_dir_name(self.log_file)
        reader = LogFileReader(path=self.log_file, workdir=log_dir)
        parser = MarianLogParser()

        writers = []
        writers.append(TensorboardWriter(log_dir))
        if self.azureml:
            writers.append(AzureMLMetricsWriter())

        first = True
        while not self.shutdown_flag.is_set():
            if first:
                logger.info(f"Processing logs for {self.log_file}")

            for line_no, line in enumerate(reader.read()):
                for log_tuple in parser.parse_line(line):
                    logger.debug(f"{self.log_file}:{line_no} produced {log_tuple}")
                    for writer in writers:
                        writer.write(*log_tuple)

            if first:
                logger.info(f"Finished processing logs for {self.log_file}")

            if self.update_freq == 0:  # just a single iteration if requested
                break

            if first:
                logger.info(
                    f"Monitoring {self.log_file} for updates every {self.update_freq}s"
                )

            time.sleep(self.update_freq)
            first = False

        logger.debug(f"Thread #{self.ident} stopped")

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

    # Create working directory if it does not exist
    if not Path(args.work_dir).exists():
        logger.warning(f"The directory '{args.work_dir}' does not exists, creating...")
        try:
            Path(args.work_dir).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(f"Insufficient permission to create {args.work_dir}")
            sys.exit(os.EX_OSFILE)

    try:
        # Create a convertion job for each log file
        jobs = []
        for log_file in args.log_file:
            if not Path(log_file).exists():
                logger.error(f"Log file not found: {log_file}")
                raise FileNotFoundError

            update_freq = 0 if args.offline else args.update_freq

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
            logger.info(f"Serving TensorBoard at https://localhost:{args.port}/")

        while True:  # Keep the main thread running so that signals are not ignored
            time.sleep(0.5)

    except ServiceExit:
        logger.info("Exiting... it may take a few seconds")
        for job in jobs:
            job.shutdown_flag.set()
        for job in jobs:
            job.join()

    except FileNotFoundError:
        sys.exit(os.EX_NOINPUT)

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
        help="TensorBoard logging directory, default: logdir",
    )
    parser.add_argument(
        "-p",
        "--port",
        help="port number for TensorBoard, default: %(default)s",
        type=int,
        default=6006,
    )
    parser.add_argument(
        "-u",
        "--update-freq",
        help="update frequency in seconds, default: %(default)s",
        type=int,
        default=UPDATE_FREQ,
    )
    parser.add_argument(
        "--offline",
        help="do not monitor for log updates, overwrites --update-freq",
        action="store_true",
    )
    parser.add_argument(
        "--azureml",
        help="generate Azure ML Metrics; updates --work-dir automatically",
        action="store_true",
    )
    parser.add_argument("--debug", help="print debug messages", action="store_true")
    parser.add_argument(
        "--version", action="version", version="%(prog)s {}".format(VERSION)
    )
    args = parser.parse_args()

    # Set logging level
    logging.getLogger("marian-tensorboard").setLevel(
        logging.DEBUG if args.debug else logging.INFO
    )

    # Set --azureml automatically if running on Azure ML
    azureml_run_id = os.getenv("AZUREML_RUN_ID", None)
    if azureml_run_id:
        args.azureml = True

    if args.azureml:
        # Try to set TensorBoard logdir to the one set on Azure ML
        if not args.work_dir:
            args.work_dir = os.getenv("AZUREML_TB_PATH", None)
        if not args.work_dir:
            args.work_dir = "/tb_logs"
        logger.info(f"AzureML RunID: {azureml_run_id}")
        logger.info(f"AzureML Setting TensorBoard logdir: {args.work_dir}")

    # Set default value for --work-dir
    if not args.work_dir:
        args.work_dir = "logdir"

    return args


if __name__ == "__main__":
    main()
