# Marian Tensorboard

TensorBoard integration for Marian NMT.

## Installation

    git clone https://github.com/marian-nmt/marian-tensorboard
    virtualenv -p python3 venv
    source ./venv/bin/activate
    pip install -r requirements.txt

## Example usage

    python3 src/marian_tensorboard/marian_tensorboard.py --log-file examples/train.encs.*.log

Open a web browser at `localhost:6006`.

## Contributors

* Amr Hendy
* Kevin Duh
* Roman Grundkiewicz
