#!/bin/bash

# Script to setup poetry environment for any job

module load miniconda3
pip install pipx
pipx install poetry
poetry install

