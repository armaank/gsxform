.DEFAULT_GOAL = help

PYTHON := python3
PIP := pip3
CONDA := conda
SHELL := bash

.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --no-builtin-rules

.PHONY: help
help:
	@printf "Usage:\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-10s\033[0m%s\n", $$1, $$2}'

.PHONY: conda
	conda:  # Set up a conda environment
	@printf "Creating conda environment...\n"
	${CONDA} env create -f env.yml
	${CONDA} activate env-gsxform
	${CONDA} deactivate

.PHONY: 
	conda-export: # Export conda environment
	@printf "Exporting conda environment...\n"
	${CONDA} env export > env.yml
