#!/bin/sh

black beta_vae_oxford_flowers

nbqa black notebooks/*.ipynb --nbqa-mutate
