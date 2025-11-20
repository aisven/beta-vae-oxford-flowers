#!/bin/sh

# lint production code
uv run ruff check beta_vae_oxford_flowers

# line code in notebook code cells
uv run ruff check notebooks/*.ipynb
