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
    num_steps: int,
    do_wandb: bool = True,
    wandb_init_config: dict = None,
) -> optax.Params:

    def compute_clause_tensor(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ) -> jnp.ndarray:
        params = jax.nn.sigmoid(params)
        x = jnp.take(params, jnp.abs(literal_tensor), fill_value=1.0, axis=1)
        x = jnp.where(literal_tensor > 0, x, 1 - x)
        return x

    # tmp = torch.concat(
    #     [
    #         torch.ones(input.shape[0], input.shape[1], 1, device=input.device),
    #         torch.cumprod(input, dim=-1)[:, :, :-1],
    #     ],
    #     dim=-1,
    # ) * torch.concat(
    #     [
    #         torch.flip(torch.cumprod(torch.flip(input, [2]), dim=-1), [2])[:, :, 1:],
    #         torch.ones(input.shape[0], input.shape[1], 1, device=input.device),
    #     ],
    #     dim=-1,
    # )

    @jax.custom_vjp
    def compute_loss(
        x: jnp.ndarray,
    ):
        rounded_x = jnp.round(x)
        # return jnp.log(jnp.sum(jnp.square(x), axis=-1) + 1e-10).sum() # TODO: USE THIS TO TURN ON OFF taking LOG of loss to boost gradient scale
        return jnp.square(jnp.prod(rounded_x, axis=-1)).sum()

    def compute_loss_fwd(x):
        return compute_loss(x), x

    def compute_loss_bwd(old_x, g):
        import pdb; pdb.set_trace()
        return old_x * g

    compute_loss.defvjp(compute_loss_fwd, compute_loss_bwd)

    def forward_pass(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        clause_tensor = compute_clause_tensor(params, literal_tensor)
        return compute_loss(clause_tensor)

    # def compute_loss_logging(
    #     params: jnp.ndarray,
    #     literal_tensor: jnp.ndarray,
    # ):
    #     params = jax.nn.sigmoid(params)
    #     x = jnp.take(params, jnp.abs(literal_tensor), fill_value=1.0, axis=1)
    #     x = jnp.where(literal_tensor > 0, x, 1 - x)
    #     x = jnp.prod(x, axis=-1)
    #     return jnp.square(x)

    @functools.partial(jax.pmap, in_axes=(0, None), axis_name="num_devices")
    def backward_pass(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        loss, grads = jax.value_and_grad(forward_pass)(params, literal_tensor)
        full_loss = compute_loss_logging(params, literal_tensor)
        return loss, grads, full_loss

    # @jax.jit
    def backprop_step(
        params: jnp.ndarray,
        opt_state: optax.OptState,
        literal_tensor: jnp.ndarray,
    ):
        l =forward_pass(params[0], literal_tensor)
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
