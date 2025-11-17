"""Utilities to Run NumPyro SVI with a delta function guide to arrive at MAP noise estimates."""

import sys
import warnings
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import optax
from jax import random
from loguru import logger
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.elbo import ELBO
from numpyro.infer.svi import SVIState

warnings.filterwarnings("ignore")
logger.disable("pint")
logger.remove()
logger.add(sys.stderr, colorize=False, enqueue=True)
logger.info(f"Using {jax.default_backend()} with {jax.local_device_count()} devices")


def setup_svi(
    model: Callable,
    guide: Callable,
    loss: ELBO | None = None,
    num_warmup_steps: int = 500,
    max_epochs: int = 5000,
    peak_lr: float = 0.01,
    gradient_clipping_val: float | None = None,
) -> SVI:
    """
    Set up Stochastic Variational Inference with AdamW optimizer and cosine decay schedule.

    Parameters
    ----------
    model : Callable
        NumPyro model function.
    guide : Callable
        NumPyro guide function (typically a delta function for MAP estimation).
    loss : ELBO or None, optional
        Evidence Lower Bound loss function. If None, uses Trace_ELBO().
        Default is None.
    num_warmup_steps : int, optional
        Number of warmup steps for learning rate schedule. Default is 500.
    max_epochs : int, optional
        Maximum number of training epochs for learning rate decay. Default is 5000.
    peak_lr : float, optional
        Peak learning rate value. Default is 0.01.
    gradient_clipping_val : float or None, optional
        Maximum global norm for gradient clipping. If None, no clipping is applied.
        Default is None.

    Returns
    -------
    SVI
        Configured NumPyro SVI object with AdamW optimizer and warmup cosine decay
        learning rate schedule.

    Notes
    -----
    The learning rate schedule starts at 0, warms up to peak_lr over num_warmup_steps,
    then decays following a cosine schedule to 1% of peak_lr over max_epochs steps.
    """
    if loss is None:
        loss = Trace_ELBO()
    # Define the learning rate schedule
    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0,
        peak_value=peak_lr,
        warmup_steps=num_warmup_steps,
        decay_steps=max_epochs,
        end_value=peak_lr * 0.01,  # Decay to 10% of the peak
    )
    npyro_optimizer = numpyro.optim.optax_to_numpyro(
        # Gradient clipping if supplied
        (
            optax.adamw(learning_rate=learning_rate_schedule)
            if gradient_clipping_val is None
            else optax.chain(
                optax.clip_by_global_norm(
                    gradient_clipping_val,
                ),
                optax.adamw(learning_rate=learning_rate_schedule),
            )
        ),
    )
    return SVI(model, guide, npyro_optimizer, loss=loss)


@partial(jax.jit, static_argnums=(0, -1))
def run_training_batch(
    svi: SVI,
    svi_state: SVIState,
    rng_key: jax.Array,
    batch_size: int,
) -> SVIState:
    """
    Run SVI parameter updates for a fixed number of steps using jax.lax.scan.

    This function is JIT-compiled for speed. The SVI object and batch_size are
    treated as static arguments (static_argnums=(0, -1)).

    Parameters
    ----------
    svi : SVI
        NumPyro SVI object containing the model, guide, and optimizer.
    svi_state : SVIState
        Current state of the SVI optimizer.
    rng_key : jax.Array
        JAX random number generator key.
    batch_size : int
        Number of SVI update steps to run.

    Returns
    -------
    SVIState
        Final SVI state after batch_size update steps.

    Notes
    -----
    Uses jax.lax.scan for efficient iteration, which is faster than a Python
    for loop inside a JIT-compiled function.
    """

    def body_fn(carry, x):
        svi_state, rng_key = carry
        rng_key, subkey = jax.random.split(rng_key)
        new_svi_state, loss = svi.update(svi_state)
        return (new_svi_state, subkey), None

    # Use lax.scan to loop `body_fn` for `batch_size` iterations
    (final_svi_state, _), _ = jax.lax.scan(
        body_fn, (svi_state, rng_key), xs=None, length=batch_size,
    )

    return final_svi_state


@partial(jax.jit, static_argnums=(0, -1))
def run_training_batch_with_diagnostics(
    svi: SVI,
    svi_state: SVIState,
    rng_key: jax.Array,
    batch_size: int,
) -> tuple[SVIState, jnp.ndarray, jnp.ndarray]:
    """
    Run SVI updates with diagnostic information (gradient norms and intermediate states).

    This function is JIT-compiled for speed. The SVI object and batch_size are
    treated as static arguments (static_argnums=(0, -1)).

    Parameters
    ----------
    svi : SVI
        NumPyro SVI object containing the model, guide, and optimizer.
    svi_state : SVIState
        Current state of the SVI optimizer.
    rng_key : jax.Array
        JAX random number generator key.
    batch_size : int
        Number of SVI update steps to run.

    Returns
    -------
    final_svi_state : SVIState
        Final SVI state after batch_size update steps.
    svi_states : jnp.ndarray
        Array of intermediate SVI states at each step.
    global_norm_grads : jnp.ndarray
        Array of global gradient norms at each step, useful for monitoring
        training stability.

    Notes
    -----
    Computing gradients explicitly at each step adds computational overhead
    compared to run_training_batch. Use this function only when diagnostic
    information is needed for monitoring or debugging purposes.
    """

    def body_fn(carry, x):
        svi_state, rng_key, _ = carry
        rng_key, subkey = jax.random.split(rng_key)
        new_svi_state, loss = svi.update(svi_state)
        global_norm_grad = optax.global_norm(
            jax.grad(svi.loss.loss, argnums=1)(
                rng_key, svi.get_params(svi_state), svi.model, svi.guide
            ),
        )
        return (new_svi_state, subkey, global_norm_grad), None

    # Use lax.scan to loop `body_fn` for `batch_size` iterations
    (final_svi_state, _, _), (svi_states, _, global_norm_grads) = jax.lax.scan(
        body_fn, (svi_state, rng_key), xs=None, length=batch_size,
    )

    return final_svi_state, svi_states, global_norm_grads


def run_svi_early_stopping(
    rng_key: jax.Array,
    svi: SVI,
    batch_size: int = 1000,
    patience: int = 3,
    max_num_batches: int = 50,
    diagnostics: bool = False,
) -> dict:
    """
    Run SVI optimization with early stopping based on loss plateau detection.

    Training proceeds in batches of optimization steps. Early stopping is triggered
    when the validation loss fails to improve by more than 1.0 for `patience`
    consecutive batches.

    Parameters
    ----------
    rng_key : jax.Array
        JAX random number generator key.
    svi : SVI
        NumPyro SVI object containing the model, guide, and optimizer.
    batch_size : int, optional
        Number of optimization steps per batch. Default is 1000.
    patience : int, optional
        Number of consecutive batches without improvement (>1.0 decrease in loss)
        before early stopping is triggered. Default is 3.
    max_num_batches : int, optional
        Maximum number of batches to run. Default is 50.
    diagnostics : bool, optional
        If True, collect gradient norms and intermediate states at each step.
        This adds computational overhead. Default is False.

    Returns
    -------
    dict
        Dictionary of optimized parameter values from the best SVI state
        (lowest validation loss).

    Notes
    -----
    The early stopping criterion requires the loss to improve by at least 1.0
    to reset the patience counter. This threshold is hardcoded and may need
    adjustment for different problem scales.

    Examples
    --------
    >>> from jax import random
    >>> rng_key = jax.Array(0)
    >>> svi = setup_svi(model, guide)
    >>> params = run_svi_early_stopping(rng_key, svi, batch_size=1000, patience=3)
    """
    svi_state = svi.init(rng_key)

    svi_states_list = []
    global_norm_grads_list = []

    best_val_loss = float("inf")
    best_svi_state = svi_state
    patience_counter = 0

    logger.info(f"Starting training with batches of {batch_size} steps.")

    final_params = None
    for batch_num in range(max_num_batches):
        if diagnostics:
            svi_state, batch_svi_states_list, batch_global_norm_grads = run_training_batch_with_diagnostics(
                svi, svi_state, rng_key, batch_size
            )
            svi_states_list.append(batch_svi_states_list)
            global_norm_grads_list.append(batch_global_norm_grads)
        else:
            svi_state = run_training_batch(svi, svi_state, rng_key, batch_size)
        current_val_loss = svi.evaluate(svi_state)
        total_steps = (batch_num + 1) * batch_size
        logger.info(
            f"Batch {batch_num + 1}/{max_num_batches} | Total steps taken: {total_steps}",
        )

        # Early stopping logic
        logger.info(f"{current_val_loss=}")
        logger.info(f"{best_val_loss=}")
        difference = current_val_loss - best_val_loss if batch_num >= 1 else -np.inf
        if difference < -1:
            logger.info(
                f"Loss improved from {best_val_loss:.4f} to {current_val_loss:.4f} {difference=}. Saving state.",
            )
            best_val_loss = current_val_loss
            best_svi_state = svi_state
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"Loss did not improve. Patience: {patience_counter}/{patience} {difference=}",
            )

            if patience_counter >= patience:
                logger.info("Early stopping triggered. Halting training.")
                break

            logger.info(f"Best loss achieved: {best_val_loss:.4f}")

            final_params = svi.get_params(best_svi_state)

    logger.info("Optimization complete.")
    # This conditional is entered if we exhaust the max training batches
    # without early stopping
    if final_params is None:
        final_params = svi.get_params(best_svi_state)

    return final_params

