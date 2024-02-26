
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


def gdsolve_verbose(
    params: optax.Params,
    optimizer: optax.GradientTransformation,
    literal_tensor: jnp.ndarray,
    original_clauses: list[list[int]],
    prng_key: jnp.ndarray,
    num_steps: int,
    do_wandb: bool = True,
    do_log_all: bool = False,
    wandb_init_config: dict = None,
) -> optax.Params:

    # def debug_loss(
    #     params: jnp.ndarray,
    #     literal_tensor: jnp.ndarray,
    # ):
    #     # params = jnp.clip(params, -3.5, 3.5)
    #     # params = jax.nn.sigmoid(2*params)
    #     params = jax.nn.sigmoid(params)
    #     x = jnp.take(params, jnp.abs(literal_tensor) - 1, fill_value=0.0, axis=1)
    #     x = jnp.where(literal_tensor > 0, 1 - x, x)
    #     x = jnp.prod(x, axis=-1)
    #     # return jnp.log(jnp.sum(jnp.square(x), axis=-1) + 1e-10).sum() # TODO: USE THIS TO TURN ON OFF taking LOG of loss to boost gradient scale
    #     return x
    
    def compute_loss(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        params = jax.nn.sigmoid(2*params)
        x = jnp.take(params, jnp.abs(literal_tensor) - 1, fill_value=0.0, axis=1)
        x = jnp.where(literal_tensor > 0, 1 - x, x)
        x = jnp.prod(x, axis=-1)
        # return jnp.log(jnp.sum(jnp.square(x), axis=-1) + 1e-10).sum() # TODO: USE THIS TO TURN ON OFF taking LOG of loss to boost gradient scale
        return jnp.square(x).sum()
    
    def compute_loss_logging(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):  
        params = jax.nn.sigmoid(2*params)
        x = jnp.take(params, jnp.abs(literal_tensor) - 1, fill_value=0.0, axis=1)
        x = jnp.where(literal_tensor > 0, 1 - x, x)
        x = jnp.prod(x, axis=-1)
        return jnp.square(x)

    @functools.partial(jax.pmap, in_axes=(0, None), axis_name="num_devices")
    def backprop_step_pmap(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        loss, grads = jax.value_and_grad(compute_loss)(params, literal_tensor)
        #full_loss = compute_loss_logging(params, literal_tensor)
        return loss, grads, 0

    def backprop_step(
        params: jnp.ndarray,
        opt_state: optax.OptState,
        literal_tensor: jnp.ndarray,
    ):
        #l = compute_loss(params[0], literal_tensor)
        loss, grads, full_loss = backprop_step_pmap(params, literal_tensor)
        # grad_coeff = np.bincount(np.abs(literal_tensor).flatten(), minlength=params.shape[-1])[1:-1]
        # grad_coeff = 1 / np.where(grad_coeff==0, 1, grad_coeff)
        # updates, opt_state = optimizer.update(grads.mean(axis=0) * grad_coeff, opt_state)
        updates, opt_state = optimizer.update(grads, opt_state)

        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads, full_loss

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

    log_dict = {"loss": [], "scaled_loss":[], "grad_norm": [], "solution_count": []}
    parameter_history = []
    loss_history = []
    solution_log_interval = 1000
    solution_count = 0
    opt_state = optimizer.init(params)
    if do_wandb:
        wandb.init(**wandb_init_config)

    batch_size = params.shape[0] * params.shape[1]
    for descent_step in range(num_steps):
        #prng_key, subkey = jax.random.split(prng_key)
        # params += jax.random.normal(subkey, params.shape) * 0.25
        params = jnp.clip(params, -3.5, 3.5)
        params, opt_state, loss_values, grads, full_loss = backprop_step(
            params=params,
            opt_state=opt_state,
            literal_tensor=literal_tensor,
        )
        
        # import pdb; pdb.set_trace()
        # if descent_step % solution_log_interval == 0:
        #     solutions = get_solutions(params, literal_tensor)
        #     solution_count = len(solutions)
            # solution_count = get_solutions(params, literal_tensor)
        # print(f"descent_step: {descent_step}, Loss: {loss_values.sum()}, GradNorm:{ float(jnp.linalg.norm(grads))}, GradVar: {float(jnp.var(grads))}")
        loss_value = float(loss_values.sum())
        scaled_loss = loss_value / float(batch_size)
        #scaled_loss = jnp.log(jnp.exp(float(loss_value) / batch_size) * batch_size)
        grad_norm = float(jnp.linalg.norm(grads))
        log_dict["loss"].append(loss_value)
        log_dict["scaled_loss"].append(scaled_loss)
        log_dict["grad_norm"].append(grad_norm)
        log_dict["solution_count"].append(solution_count)
        if do_wandb:
            wandb.log(
                {
                    "loss": loss_value,
                    "scaled_loss": scaled_loss,
                    "grad_norm": grad_norm,
                    "solution_count": solution_count,
                }
            )
        if do_log_all:
            parameter_history.append(jax.nn.sigmoid(params[:, 0, :]))
            loss_history.append(full_loss[0])

    if do_wandb:
        wandb.finish()
    if do_log_all:
        all_losses = np.stack(loss_history)
        all_params = np.stack(parameter_history)
    else:
        all_losses = None
        all_params = None
    solutions = get_solutions(params, literal_tensor)
    return (
        params,
        descent_step + 1,
        loss_value,
        0.0,
        solutions,
        log_dict,
        all_params,
        all_losses,
    )
