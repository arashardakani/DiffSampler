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
):  
    num_devices = jax.local_device_count()
    # if num_devices == 1:
    #     var_embedding = jax.random.normal(key, (batch_size, cnf_problem.nv))
    # else:
    var_embedding = (
        jax.random.normal(key, (num_devices, batch_size, cnf_problem.nv))
    )
    
    # find the maximum clause length
    max_clause_len = max([len(clause) for clause in cnf_problem.clauses])
    # pad the clauses to be of the same length
    literal_tensor = jnp.array(
        [
            clause + [cnf_problem.nv + 1] * (max_clause_len - len(clause))
            for clause in cnf_problem.clauses
        ]
    )
    single_clause_rows = jnp.sum(literal_tensor != (cnf_problem.nv + 1), axis=1) == 1
    gradient_mask = jnp.isin(jnp.arange(cnf_problem.nv), literal_tensor[:, 0][single_clause_rows]).astype(jnp.float32)
    
    # broadcast mask to the shape of var_embedding
    broadcast_mask = gradient_mask.reshape(1, 1, len(gradient_mask))

    # fix variable assignments to -1 or 1 based on gradient_mask
    var_embedding = jnp.where(
        broadcast_mask == 0.,
        var_embedding,
        jnp.broadcast_to(broadcast_mask, var_embedding.shape)
    )
    if optimizer_str == "adamw":
        optimizer = optax.adamw(learning_rate=learning_rate)
    elif optimizer_str == "adam":
        optimizer = optax.adam(learning_rate=learning_rate)
    elif optimizer_str == "sgd":
        optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
    else:
        raise NotImplementedError
    return var_embedding, optimizer, literal_tensor, gradient_mask