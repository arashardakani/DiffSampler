from typing import Callable
import logging
import functools
import time

from pysat.formula import CNF
import jax.numpy as jnp
import jax
import numpy as np
import optax
from tqdm import tqdm
import wandb


def gdsolve(
    params: optax.Params,
    optimizer: optax.GradientTransformation,
    literal_tensor: jnp.ndarray,
    num_steps: int,
) -> optax.Params:
    """for runtime measurement"""

    @functools.partial(jax.pmap, in_axes=(0, None))
    def scan_sat_solutions(
        assignment: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        sat = jnp.take(assignment, jnp.abs(literal_tensor)-1, fill_value=0, axis=1)
        sat = jnp.where(literal_tensor > 0, sat, 1-sat)
        sat = jnp.all(jnp.any(sat > 0, axis=2), axis=1)
        return sat
    
    def get_solutions(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        assignment = (jax.nn.sigmoid(params) > 0.5).astype(int)
        solution_mask = scan_sat_solutions(assignment, literal_tensor)
        solutions = assignment[solution_mask]
        solutions = solutions.reshape(-1, solutions.shape[-1])
        solutions = np.unique(solutions, axis=0)
        return solutions

    def compute_loss(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        params = jax.nn.sigmoid(2*params)
        x = jnp.take(params, jnp.abs(literal_tensor) - 1, fill_value=0.0, axis=1)
        x = jnp.where(literal_tensor > 0, 1 - x, x)
        x = jnp.prod(x, axis=-1)
        return jnp.square(x).sum(axis=-1).sum()

    @functools.partial(jax.pmap, in_axes=(0, None), axis_name="num_devices")
    def backward_pass(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        loss, grads = jax.value_and_grad(compute_loss)(params, literal_tensor)
        return loss, grads

    def backprop_step(
        params: jnp.ndarray,
        opt_state: optax.OptState,
        literal_tensor: jnp.ndarray,
    ):
        loss_value, grads = backward_pass(params, literal_tensor)
        updates, opt_state = optimizer.update(grads.mean(axis=0), opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value.sum()

    start_t = time.time()
    opt_state = optimizer.init(params)
    for step in range(num_steps):
        params = jnp.clip(params, -3.5, 3.5)
        params, opt_state, loss_value = backprop_step(
            params=params,
            opt_state=opt_state,
            literal_tensor=literal_tensor,
        )
        # print(step, loss_value)
    end_t = time.time()
    solutions = get_solutions(params, literal_tensor)
    return params, step + 1, loss_value, end_t - start_t, solutions
