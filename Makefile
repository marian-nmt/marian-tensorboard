SHELL := /bin/bash

all: init install test

init:
	test -d venv || virtualenv -p python3 venv
	source ./venv/bin/activate

install: requirements.txt
	python3 -m pip install -r $<

test:
	pytest

build:
	python3 -m pip install --upgrade build
	python3 -m build
	python3 -m pip install --upgrade twine
	## Upload to TestPyPI
	# python3 -m twine upload --repository testpypi dist/*
	## Install from test repository
	# python3 -m venv env
	# source env/bin/activate
	# (env) python3 -m pip install --index-url https://test.pypi.org/simple/ marian-tensorboard

clean:
	rm -rf build/ dist/ tests/__pycache__/ src/marian_tensorboard.egg-info/ src/marian_tensorboard/__pycache__/
