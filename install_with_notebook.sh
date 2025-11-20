#!/bin/sh

# install project with dependencies plus dev and test plus notebook dependencies
pip install .[dev,test,notebook]

# remove the project itself to resemble a --no-root or --no-project behavior
pip uninstall -y beta-vae-oxford-flowers
