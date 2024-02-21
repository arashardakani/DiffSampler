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
    learning_rate: float = 1.0,
    optimizer_str: str = "sgd",
):
    
    start_time = time.time()
    num_devices = jax.local_device_count()
    # if num_devices == 1:
    #     var_embedding = jax.random.normal(key, (batch_size, cnf_problem.nv))
    # else:

    var_embedding = (
        jax.random.normal(key, (num_devices, batch_size, cnf_problem.nv))
    )
    # var_embedding = var_embedding.at[0].set([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # var_embedding = var_embedding[None,:,:]
    # find the maximum clause length
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
    print(f"Time taken to initialize problem: {end_time - start_time} seconds")

    start_time = time.time()
    all_excluded_vars = []
    while True:
        single_clause_vars = literal_tensor[np.expand_dims((np.abs(literal_tensor) > 0).sum(axis=-1) == 1, axis=1 ) & (np.abs(literal_tensor) > 0) ]
        all_excluded_vars.extend(single_clause_vars)
        if len(single_clause_vars) == 0:
            break
        literal_tensor = np.delete(literal_tensor, np.argwhere(np.isin(literal_tensor, single_clause_vars).sum(axis=-1) > 0), 0)
        literal_tensor[np.isin(literal_tensor, -1 * single_clause_vars)] = 0
    # swap all 0's to fill_value
    literal_tensor = np.where(literal_tensor == 0, fill_value, literal_tensor)
    print(literal_tensor.shape)
    end_time = time.time()
    print(f"Time taken to prune problem: {end_time - start_time} seconds")
    # learning rate heuristic
#    learning_rate = 0.1*(np.log(cnf_problem.nv * len(cnf_problem.clauses)))
 
    if optimizer_str == "adamw":
        optimizer = optax.adamw(learning_rate=learning_rate)
    elif optimizer_str == "adam":
        optimizer = optax.adam(learning_rate=learning_rate)
    elif optimizer_str == "sgd":
        optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.0)
    else:
        raise NotImplementedError
    return var_embedding, optimizer, literal_tensor
