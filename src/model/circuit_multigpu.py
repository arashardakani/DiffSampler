from pysat.formula import CNF
import jax.numpy as jnp
import jax
import numpy as np
import optax
import time
from tqdm import tqdm
from typing import Callable
import logging
import functools


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
        var_embedding = jax.random.normal(key, (num_devices, batch_size, cnf_problem.nv))
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
    import pdb; pdb
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
        satisfying_row_indices = jnp.where(sat, jnp.arange(sat.shape[0]), sat.shape[0]+1)
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
        pruned_solutions = jnp.take(solutions, jnp.where(jnp.any(solutions >= 0, axis=1))[0], axis=0)
        return np.unique(pruned_solutions, axis=1)


    def compute_loss(
        params: jnp.ndarray,
    ):  
        params = jax.nn.sigmoid(params)
        x = jnp.take(params, jnp.abs(literal_tensor), fill_value=1.0, axis=1)
        x = jnp.where(literal_tensor > 0, x, 1 - x)
        x = 1 - jnp.prod(x, axis=-1)
        labels = jnp.ones((x.shape[-2], x.shape[-1]))        
        return optax.l2_loss(x, labels).sum()

    @functools.partial(jax.pmap, in_axes=(0, None), axis_name='num_devices')
    def backward_pass(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        loss, grads = jax.value_and_grad(compute_loss)(params)
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
) -> optax.Params:

    def compute_loss(
        params: jnp.ndarray,
    ):  
        params = jax.nn.sigmoid(params)
        x = jnp.take(params, jnp.abs(literal_tensor), fill_value=1.0, axis=1)
        x = jnp.where(literal_tensor > 0, x, 1 - x)
        x = 1 - jnp.prod(x, axis=-1)
        labels = jnp.ones((x.shape[-2], x.shape[-1]))        
        return optax.l2_loss(x, labels).sum()

    @functools.partial(jax.pmap, in_axes=(0, None), axis_name='num_devices')
    def backward_pass(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        loss, grads = jax.value_and_grad(compute_loss)(params)
        return loss, grads

    # @jax.jit
    def backprop_step(
        params: jnp.ndarray,
        opt_state: optax.OptState,
        literal_tensor: jnp.ndarray,
    ):
        loss_value, grads = backward_pass(params, literal_tensor)
        updates, opt_state = optimizer.update(grads.mean(axis=0), opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value.sum(), grads
    
    @functools.partial(jax.pmap, in_axes=(0, None))
    def scan_sat_solutions(
        assignment: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):  
        sat = jnp.take(assignment, jnp.abs(literal_tensor), fill_value=1, axis=1)
        sat = jnp.where(literal_tensor > 0, 1 - sat, sat)
        sat = jnp.all(jnp.any(sat > 0, axis=2), axis=1)
        satisfying_row_indices = jnp.where(sat, jnp.arange(sat.shape[0]), sat.shape[0]+1)
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
        pruned_solutions = jnp.take(solutions, jnp.where(jnp.any(solutions >= 0, axis=1))[0], axis=0)
        return np.unique(pruned_solutions, axis=1)
            
    # literal_tensor_sharded = jnp.stack([literal_tensor]*num_devices)
    log_dict = {'loss': [], 'grad_norm': [], 'solution_count': []}
    solution_log_interval = 5
    solution_count = 0
    # opt_state = optimizer.init(params)
    opt_state = optimizer.init(params)
    for step in tqdm(range(num_steps), desc="Gradient Descent"):
        params, opt_state, loss_value, grads = backprop_step(
            params=params,
            opt_state=opt_state,
            literal_tensor=literal_tensor,
        )
        log_dict['loss'].append(float(loss_value))
        log_dict['grad_norm'].append(float(jnp.linalg.norm(grads)))
        if step % solution_log_interval == 0:
            solutions = get_solutions(params, literal_tensor)
            solution_count = len(solutions)
        log_dict['solution_count'].append(solution_count)
        # logging.info(f"Step {step}, loss: {loss_value}, grad_norm: {jnp.linalg.norm(grads)}, solution_count: {len(get_solutions(params, literal_tensor))}")
        # logging.info(f"{jax.nn.sigmoid(params[:5])}")
        # if step == 10:
        #     import pdb; pdb.set_trace()
    solutions = solutions = get_solutions(params, literal_tensor)
    return params, step + 1, loss_value, 0.0, solutions, log_dict