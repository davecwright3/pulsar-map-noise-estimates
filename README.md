# MAP Noise Estimates with Discovery and NumPyro

Estimate noise parameters in pulsar timing data.

## Overview

This package provides tools to:

1. **Estimate noise parameters** using Stochastic Variational Inference (SVI) with NumPyro
   - White noise parameters (EFAC, EQUAD, ECORR)
   - Red noise (power-law spectral models)
   - Dispersion measure (DM) noise
   - Chromatic noise
   - Solar wind effects

2. **Add estimated parameters to PINT timing models** for pulsar timing analysis

The optimization uses JAX for efficient GPU/CPU acceleration with JIT compilation, AdamW optimization, and early stopping based on loss plateau detection.

## Installation

This project uses [pixi](https://pixi.sh) for dependency management:

```bash
pixi install
```

However, I've provided a conda `environment.yml` exported from pixi if you'd like to use conda.

## Usage

Below, I assume you have configured a NumPyro `model` using a discovery likelihood and a PINT timing model.

```python
from jax import random
from pulsar_map_noise_estimates.map_noise_estimate import setup_svi, run_svi_early_stopping
from pulsar_map_noise_estimates.add_to_model import add_noise_to_model

# Set up and run SVI optimization
max_num_samples = 50_000
batch_size = 1_000

rng_key = random.key(117)

guide = numpyro.infer.autoguide.AutoDelta(model)
svi = setup_svi(model, guide, max_epochs=max_num_samples, num_warmup_steps=batch_size)
params = run_svi_early_stopping(
    rng_key,
    svi,
    batch_size = batch_size,
    max_num_batches = max_num_samples // batch_size
    )
# Post-process params
# I don't do it manually because you may choose a different guide and the output could be different
params  = {key.removesuffix("_auto_loc"): value for key, value in params.items()}
# Add estimated noise parameters to timing model
updated_model = add_noise_to_model(timing_model, params)
```
