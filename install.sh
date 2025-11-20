#!/bin/sh

# ensure that the virtual environment
# exists and is stored in subdirectory .venv
uv venv .venv

# activate the virtual environment
. .venv/bin/activate

# ensure that the dependencies in the virtual environment
# are aligned with the definitions in pyproject.toml
# including also dev and test dependencies
# thereby avoid installing the project itself into the environment
uv sync --extra dev --extra test --no-install-project
