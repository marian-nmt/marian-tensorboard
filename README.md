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
TensorBoard charts every `--update-freq` seconds unless `--offline` is used.

### Azure ML

    marian-tensorboard -f path/to/train.log [-t tb azureml]

Then on Azure Machine Learning VM go to the __Metrics__ tab or start a
TensorBoard server under the __Endpoints__ tab.

Note that logging into Azure ML Metrics is automatically enabled if Azure ML
Run ID is detected. Specify `-t azureml` to disable TensorBoard logging.
If Azure ML is enabled, the script will not start an own TensorBoard server
instance.

## Contributors

* Amr Hendy
* Kevin Duh
* Roman Grundkiewicz
* Marcin Junczys-Dowmunt

See [CHANGELOG.md](CHANGELOG.md).

## License

See [LICENSE.md](LICENSE.md).
