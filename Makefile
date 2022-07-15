.DEFAULT_GOAL = help

.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --no-builtin-rules

PYTHON := python3
PIP := pip3
SHELL := bash
CONDA := $(conda info --base)

OS := $(uname -s)
ifeq ($(OS),Darwin)        # Mac OS X
	CONDA_ENV := env_osx.yml
endif
ifeq ($(OS),Linux)
	CONDA_ENV := env_linux.yml
endif

CONDA_ACTIVATE = source $(CONDA)/etc/profile.d/conda.sh ; $(CONDA) activate ; $(CONDA) activate

.PHONY: help
help:
	@printf "Usage:\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-10s\033[0m%s\n", $$1, $$2}'

.PHONY: conda
conda: # Setup conda environment
	@printf "Creating conda environment...\n"
	#${CONDA} config --set restore_free_channel true
	$(conda info --base) env create -f $(CONDA_ENV)

.PHONY: export-conda
export-conda: # Export conda environment
	@printf "Exporting conda environment...\n"
	$(CONDA) env export --no-builds > $(CONDA_ENV)

.PHONY: setup
setup: # Setup dev environment 
	@printf "Setting up dev environment...\n"
	$(CONDA_ACTIVATE) env-gsxform
	@printf "Activated conda environment...\n"
	pre-commit install
	@printf "Setup pre-commit hooks...\n"

.PHONY: export-pip
export-pip: # Export pip environment
	@printf "Exporting pip environment...\n"
	$(PIP) list --format=freeze > requirements.txt
