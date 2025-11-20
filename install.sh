#!/bin/sh

# pip install .
# also install dev and test dependencies
pip install .[dev,test]

# remove the project itself to resemble a --no-root or --no-project behavior
pip uninstall -y beta-vae-oxford-flowers
