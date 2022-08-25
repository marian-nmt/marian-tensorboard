SHELL := /bin/bash

.PHONY: all build clean install test

all: install test

# Setup new Python virtual environment
venv:
	test -d $@ || virtualenv -p python3 $@

# Install requirements
install: requirements.txt venv
	source venv/bin/activate && pip install -r $<
	source venv/bin/activate && pip install --upgrade requests

# Run unit tests
test: venv
	source venv/bin/activate && pytest

# Clean temporary files
clean:
	rm -rf build/ dist/ tests/__pycache__/ src/marian_tensorboard.egg-info/ src/marian_tensorboard/__pycache__/

# This mostly serves as a reminder of how to publish a new release to PyPI
build:
	python3 -m pip install --upgrade build twine
	python3 -m build
	## Upload to TestPyPI
	# python3 -m twine upload --repository testpypi dist/*
	## Install from test repository
	# python3 -m venv env
	# source env/bin/activate
	# (env) python3 -m pip install --index-url https://test.pypi.org/simple/ marian-tensorboard
	## Upload to PyPI
	# twine upload dist/*
