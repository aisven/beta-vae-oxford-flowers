#!/bin/sh

# format production code
black beta_vae_oxford_flowers

# format code in notebook code cells
nbqa black notebooks/*.ipynb --nbqa-mutate
