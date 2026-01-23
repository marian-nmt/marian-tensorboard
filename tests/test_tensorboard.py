#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import socket
import tempfile
import time

import pytest

from marian_tensorboard.marian_tensorboard import (
    ConversionJob,
    launch_tensorboard,
)

# Path to examples directory
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


def is_port_open(host, port, timeout=1):
    """Check if a port is open on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False


def test_marian_tensorboard():
    """Test starting TensorBoard server with example log files."""
    # Use example log files
    log_file_1 = os.path.join(EXAMPLES_DIR, "train.encs.1.log")
    log_file_2 = os.path.join(EXAMPLES_DIR, "train.encs.2.log")

    assert os.path.exists(log_file_1), f"Log file not found: {log_file_1}"
    assert os.path.exists(log_file_2), f"Log file not found: {log_file_2}"

    # Create a temporary directory for TensorBoard logs
    work_dir = tempfile.mkdtemp(prefix="marian_tb_test_")

    try:
        # Create and start conversion jobs for each log file (offline mode)
        jobs = []
        for log_file in [log_file_1, log_file_2]:
            job = ConversionJob(
                log_file,
                work_dir,
                update_freq=0,  # Offline mode: single iteration
                step="updates",
                tb=True,
                azureml=False,
                mlflow=False,
            )
            job.start()
            jobs.append(job)

        # Wait for all jobs to complete
        for job in jobs:
            job.join(timeout=60)

        # Use a non-standard port to avoid conflicts
        test_port = 16006

        # Launch TensorBoard server
        launch_tensorboard(work_dir, test_port)

        # Wait for the server to start
        max_wait = 10  # seconds
        start_time = time.time()
        server_started = False

        while time.time() - start_time < max_wait:
            if is_port_open("localhost", test_port):
                server_started = True
                break
            time.sleep(0.5)

        assert server_started, f"TensorBoard server did not start on port {test_port}"

    finally:
        # Clean up the temporary directory
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
