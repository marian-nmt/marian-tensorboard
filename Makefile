SHELL := /bin/bash

all: init install test

init:
	test -d venv || virtualenv -p python3 venv
	source ./venv/bin/activate

install: requirements.txt
	python3 -m pip install -r $<

test:
	pytest

clean:
	rm -rf build/ dist/ tests/__pycache__/ src/marian_tensorboard.egg-info/ src/marian_tensorboard/__pycache__/
