.DEFAULT_GOAL = help

PYTHON := python3
PIP := pip3
SHELL := bash
CONDA := conda

.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --no-builtin-rules

CONDA_BASE := $(shell conda info --base)

OS := $(shell uname -s)

ifeq ($(OS),Darwin)        # Mac OS X
	CONDA_ENV := env_osx.yml
endif
ifeq ($(OS),Linux)
	CONDA_ENV := env_linux.yml
endif

.PHONY: help
help:
	@printf "Usage:\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-10s\033[0m%s\n", $$1, $$2}'

.PHONY: conda
conda: # Setup conda environment
	@printf "Creating conda environment...\n"
	$(CONDA) env create -f $(CONDA_ENV)

.PHONY: export-conda
export-conda: # Export conda environment
	@printf "Exporting conda environment...\n"
	$(CONDA) env export --no-builds > $(CONDA_ENV)

.PHONY: export-pip
export-pip: # Export pip environment
	@printf "Exporting pip environment...\n"
	$(PIP) list --format=freeze > requirements.txt

.PHONY: tests
tests: # run tests
	python -m pytest -v gsxform tests --doctest-modules --html=report.html --self-contained-html --cov=./ --cov-report=xml

.PHONY: clean
clean: # clean project directory
	rm report.html coverage.xml
	rm -rf __pycache__/ ./gsxform/__pycache__/ ./tests/____pycache__/ 
