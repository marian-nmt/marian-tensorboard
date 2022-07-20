# Marian Tensorboard

TensorBoard integration for Marian NMT.

## Installation

    virtualenv -p python3 venv
    source ./venv/bin/activate
    pip install -r requirements.txt

## Example usage

    ./marian-visualize.py --log-file logs/train.encs.*.log

Open a web browser at `localhost:6006`.
