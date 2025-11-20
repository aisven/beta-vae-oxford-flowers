#!/bin/sh

# format production code
uv run black beta_vae_oxford_flowers

# format code in notebook code cells
uv run nbqa black notebooks/*.ipynb --nbqa-mutate
