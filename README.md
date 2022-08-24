# Marian Tensorboard

TensorBoard integration for [Marian NMT](https://marian-nmt.github.io/).
`marian-tensorboard` generates charts for TensorBoard or Azure ML Metrics from
Marian's training logs.

It started as a project at [MTMA 2022](https://mtma22.github.io/) and
conceptually at [MTM 2019](https://www.statmt.org/mtm19/).

## Installation

Using PyPI:

    pip install marian-tensorboard

Locally:

    git clone https://github.com/marian-nmt/marian-tensorboard
    cd marian-tensorboard
    virtualenv -p python3 venv
    source ./venv/bin/activate
    python3 setup.py install

Both will add new `marian-tensorboard` command.

## Usage

### Local machine

    marian-tensorboard -f examples/train.encs.*.log

Open a web browser at `https://localhost:6006`. The script will update the
TensorBoard charts every 5 seconds unless `--offline` is used.

### Azure ML

    marian-tensorboard -f path/to/train.log [-t tb azureml]

Then on Azure Machine Learning VM go to the __Metrics__ tab or start a
TensorBoard server under the __Endpoints__ tab. Using `-t tb azureml` will try
to set `--work-dir` automatically for the TensorBoard that is run internally at
Azure ML and prevent the script from starting own instance.

## Contributors

* Amr Hendy
* Kevin Duh
* Roman Grundkiewicz

See [CHANGELOG.md](CHANGELOG.md).

## License

See [LICENSE.md](LICENSE.md).
