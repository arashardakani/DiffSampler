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


def init_problem(
    cnf_problem: CNF,
    batch_size: int,
    key: jnp.ndarray = None,
    learning_rate: float = 1.0,
    optimizer_str: str = "sgd",
    single_device: bool = False,
):
    if single_device:
        var_embedding = jax.random.normal(key, (batch_size, cnf_problem.nv))
    else:
        num_devices = jax.local_device_count()
        var_embedding = (
            jax.random.normal(key, (num_devices, batch_size, cnf_problem.nv)) * 0.1
        )
    max_clause_len = max([len(clause) for clause in cnf_problem.clauses])
    num_clauses = len(cnf_problem.clauses)
    literal_tensor = jnp.array(
        [
            [c + (-1) ** (c > 0) for c in clause]
            + [num_clauses] * (max_clause_len - len(clause))
            for clause in cnf_problem.clauses
        ]
    )
    if optimizer_str == "adamw":
        optimizer = optax.adamw(learning_rate=learning_rate)
    elif optimizer_str == "adam":
        optimizer = optax.adam(learning_rate=learning_rate)
    elif optimizer_str == "sgd":
        optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
    else:
        raise NotImplementedError
    return var_embedding, optimizer, literal_tensor


def run_back_prop(
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
        sat = jnp.take(assignment, jnp.abs(literal_tensor), fill_value=1, axis=1)
        sat = jnp.where(literal_tensor > 0, 1 - sat, sat)
        sat = jnp.all(jnp.any(sat > 0, axis=2), axis=1)
        satisfying_row_indices = jnp.where(
            sat, jnp.arange(sat.shape[0]), sat.shape[0] + 1
        )
        return jnp.take(assignment, satisfying_row_indices, axis=0, fill_value=-1)

    def get_solutions(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        assignment = (jax.nn.sigmoid(params) > 0.5).astype(int)
        if assignment.ndim == 2:
            assignment = jnp.expand_dims(assignment, axis=0)
        solutions = scan_sat_solutions(assignment, literal_tensor)
        solutions = solutions.reshape((-1, solutions.shape[-1]))
        # remove spurious solutions that are all -1's
        pruned_solutions = jnp.take(
            solutions, jnp.where(jnp.any(solutions >= 0, axis=1))[0], axis=0
        )
        return np.unique(pruned_solutions, axis=1)

    def compute_loss(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        params = jax.nn.sigmoid(params)
        x = jnp.take(params, jnp.abs(literal_tensor), fill_value=1.0, axis=1)
        x = jnp.where(literal_tensor > 0, x, 1 - x)
        x = jnp.prod(x, axis=-1)
        return jnp.square(x).sum()

    @functools.partial(jax.pmap, in_axes=(0, None), axis_name="num_devices")
    def backward_pass(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        loss, grads = jax.value_and_grad(compute_loss)(params, literal_tensor)
        return loss, grads

    @jax.jit
    def backprop_step(
        params: jnp.ndarray,
        opt_state: optax.OptState,
        literal_tensor: jnp.ndarray,
    ):
        loss_value, grads = backward_pass(params, literal_tensor)
        updates, opt_state = optimizer.update(grads.mean(axis=0), opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value.sum(), grads

    start_t = time.time()
    opt_state = optimizer.init(params)
    for step in range(num_steps):
        params, opt_state, loss_value = backprop_step(
            params=params,
            opt_state=opt_state,
            literal_tensor=literal_tensor,
        )
    end_t = time.time()
    solutions = jnp.unique(get_solutions(params, literal_tensor), axis=1)
    return params, step + 1, loss_value, end_t - start_t, solutions


def run_back_prop_verbose(
    params: optax.Params,
    optimizer: optax.GradientTransformation,
    literal_tensor: jnp.ndarray,
    num_steps: int,
    do_wandb: bool = True,
    wandb_init_config: dict = None,
) -> optax.Params:
    def compute_loss(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        x = jnp.take(params, jnp.abs(literal_tensor), fill_value=1.0, axis=1)
        x = jnp.where(literal_tensor > 0, x, 1 - x)
        x = jnp.prod(x, axis=-1)
        # return jnp.log(jnp.sum(jnp.square(x), axis=-1) + 1e-10).sum() # TODO: USE THIS TO TURN ON OFF taking LOG of loss to boost gradient scale
        return jnp.square(x).sum()

    def compute_loss_logging(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        x = jnp.take(params, jnp.abs(literal_tensor), fill_value=1.0, axis=1)
        x = jnp.where(literal_tensor > 0, x, 1 - x)
        x = jnp.prod(x, axis=-1)
        # return jnp.log(jnp.sum(jnp.square(x), axis=-1) + 1e-10).sum() # TODO: USE THIS TO TURN ON OFF taking LOG of loss to boost gradient scale
        return jnp.square(x)

    @functools.partial(jax.pmap, in_axes=(0, None), axis_name="num_devices")
    def backward_pass(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        params = jax.nn.sigmoid(params)
        loss, grads = jax.value_and_grad(compute_loss)(params, literal_tensor)
        full_loss = compute_loss_logging(params, literal_tensor)
        return loss, grads, full_loss

    # @jax.jit
    def backprop_step(
        params: jnp.ndarray,
        opt_state: optax.OptState,
        literal_tensor: jnp.ndarray,
    ):
        # l,g = compute_loss(params[0], literal_tensor)
        loss, grads, full_loss = backward_pass(params, literal_tensor)
        updates, opt_state = optimizer.update(grads.mean(axis=0), opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads, full_loss

    @functools.partial(jax.pmap, in_axes=(0, None))
    def scan_sat_solutions(
        assignment: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        sat = jnp.take(assignment, jnp.abs(literal_tensor), fill_value=1, axis=1)
        sat = jnp.where(literal_tensor > 0, 1 - sat, sat)
        sat = jnp.all(jnp.any(sat > 0, axis=2), axis=1)
        satisfying_row_indices = jnp.where(
            sat, jnp.arange(sat.shape[0]), sat.shape[0] + 1
        )
        return jnp.take(assignment, satisfying_row_indices, axis=0, fill_value=-1)

    def get_solutions(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        assignment = (jax.nn.sigmoid(params) > 0.5).astype(int)
        if assignment.ndim == 2:
            assignment = jnp.expand_dims(assignment, axis=0)
        solutions = scan_sat_solutions(assignment, literal_tensor)
        solutions = solutions.reshape((-1, solutions.shape[-1]))
        # remove spurious solutions that are all -1's
        pruned_solutions = jnp.take(
            solutions, jnp.where(jnp.any(solutions >= 0, axis=1))[0], axis=0
        )
        return np.unique(pruned_solutions, axis=1)

    # literal_tensor_sharded = jnp.stack([literal_tensor]*num_devices)
    log_dict = {"loss": [], "scaled_loss": [], "grad_norm": [], "solution_count": []}
    parameter_history = []
    loss_history = []
    solution_log_interval = 5
    solution_count = 0
    batch_size = params.shape[1] * params.shape[0]
    opt_state = optimizer.init(params)
    if do_wandb:
        wandb.init(**wandb_init_config)
    for step in tqdm(range(num_steps), desc="Gradient Descent"):
        params, opt_state, loss_values, grads, full_loss = backprop_step(
            params=params,
            opt_state=opt_state,
            literal_tensor=literal_tensor,
        )
        if step % solution_log_interval == 0:
            solutions = get_solutions(params, literal_tensor)
            solution_count = len(solutions)

        loss_value = loss_values.sum()
        scaled_loss = jnp.log(jnp.exp(float(loss_value) / batch_size) * batch_size)
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
        parameter_history.append(jax.nn.sigmoid(params[:, 0, :]))
        loss_history.append(full_loss[0])
    if do_wandb:
        wandb.finish()
    all_losses = np.stack(loss_history)
    all_params = np.stack(parameter_history)
    solutions = solutions = get_solutions(params, literal_tensor)
    return (
        params,
        step + 1,
        loss_value,
        0.0,
        solutions,
        log_dict,
        all_params,
        all_losses,
    )
