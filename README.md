# Marian Tensorboard

TensorBoard integration for Marian NMT.

## Installation

Using PyPI:

    python3 -m pip install marian-tensorboard

Locally:

    git clone https://github.com/marian-nmt/marian-tensorboard
    virtualenv -p python3 venv
    source ./venv/bin/activate
    pip install -r requirements.txt

## Usage

### Local machine

    python3 marian-tensorboard --log-file examples/train.encs.*.log

Open a web browser at `localhost:6006`.

### Azure ML

    python3 marian-tensorboard --azureml

On Azure Machine Learning VM go to the __Metrics__ tab or start a TensorBoard
server on the __Endpoints__ tab.

## License

See LICENSE.md.

## Contributors

* Amr Hendy
* Kevin Duh
* Roman Grundkiewicz

See CHANGELOG.md.
