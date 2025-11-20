#!/bin/sh

# lint production code
ruff check beta_vae_oxford_flowers

# line code in notebook code cells
ruff check notebooks/*.ipynb
