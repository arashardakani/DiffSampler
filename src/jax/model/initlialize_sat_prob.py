from typing import Callable
import logging
import functools
import time

from pysat.formula import CNF
import jax.numpy as jnp
import numpy as np
import time


import jax
import numpy as np
import optax
from tqdm import tqdm
import wandb

# @jax.jit
# def prune_step(
#     tensor: jnp.ndarray,
#     fill_value: int = 0,
# ):
#     single_clause_vars = tensor[jnp.expand_dims((jnp.abs(tensor) > 0).sum(axis=-1) == 1, axis=1 ) & (jnp.abs(tensor) > 0) ]
#     tensor = jnp.delete(tensor, jnp.argwhere(jnp.isin(tensor, single_clause_vars).sum(axis=-1) > 0), 0)
#     tensor[jnp.isin(tensor, -1 * single_clause_vars)] = 0
#     return tensor, single_clause_vars

# def prune_problem(
#     literal_tensor: jnp.ndarray,
#     fill_value: int = 0,
# ):
#     while True:
#         literal_tensor, changed = prune_step(literal_tensor, fill_value)
#         if not changed:
#             break
#     return literal_tensor

def init_problem(
    cnf_problem: CNF,
    batch_size: int,
    key: jnp.ndarray = None,
):
    
    start_time = time.time()
    num_devices = jax.local_device_count()

    key, subkey = jax.random.split(key)
    var_embedding = (
        jax.random.normal(key, (num_devices, batch_size, cnf_problem.nv))
    )
    max_clause_len = max([len(clause) for clause in cnf_problem.clauses])
    fill_value = cnf_problem.nv + 1
    # pad the clauses to be of the same length
    literal_tensor = np.array(
        [
            clause + [0] * (max_clause_len - len(clause))
            for clause in cnf_problem.clauses
        ]
    )
    end_time = time.time()
    # print(f"Time taken to initialize problem: {end_time - start_time} seconds")

    start_time = time.time()
    all_excluded_vars = []
    while True:
        single_clause_vars = literal_tensor[np.expand_dims((np.abs(literal_tensor) > 0).sum(axis=-1) == 1, axis=1 ) & (np.abs(literal_tensor) > 0) ]
        all_excluded_vars.extend(single_clause_vars)
        if len(single_clause_vars) == 0:
            break
        literal_tensor = np.delete(literal_tensor, np.argwhere(np.isin(literal_tensor, single_clause_vars).sum(axis=-1) > 0), 0)
        literal_tensor[np.isin(literal_tensor, -1 * single_clause_vars)] = 0
    
    # for each excluded var, fill values in var_embedding to according 0 or 1 values
    for var in all_excluded_vars:
        abs_var = np.abs(var)
        var_embedding = var_embedding.at[:,:, abs_var-1].set(3.5 - float(var < 0) * 7.0)
    
    # swap all 0's to fill_value
    literal_tensor = np.where(literal_tensor == 0, fill_value, literal_tensor)
    end_time = time.time()
    # print(f"Time taken to prune problem: {end_time - start_time} seconds")
    # print(literal_tensor.shape)
    # print(all_excluded_vars)
    return var_embedding, literal_tensor, key

def init_optimizer(
    optimizer_str: str,
    learning_rate: float,
    momentum: str = "0.0",
):
    if optimizer_str == "adamw":
        optimizer = optax.adamw(learning_rate=learning_rate)
    elif optimizer_str == "adam":
        b1 = momentum.split(",")[0]
        b2 = momentum.split(",")[1]
        optimizer = optax.adam(learning_rate=learning_rate, b1=float(b1), b2=float(b2))
    elif optimizer_str == "sgd":
        optimizer = optax.sgd(learning_rate=learning_rate, momentum=float(momentum))
    else:
        raise NotImplementedError
    return optimizer
