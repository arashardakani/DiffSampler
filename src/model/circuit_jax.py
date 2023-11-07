from pysat.formula import CNF
import jax.numpy as jnp
import jax
import optax
import time
from tqdm import tqdm
from typing import Callable
import logging


def init_problem(
    cnf_problem: CNF,
    batch_size: int,
    key: jnp.ndarray = None,
    learning_rate: float = 1.0,
    optimizer_str: str = "sgd",
):
    max_clause_len = max([len(clause) for clause in cnf_problem.clauses])
    num_clauses = len(cnf_problem.clauses)
    embedding = jax.random.normal(key, (batch_size, cnf_problem.nv))
    literal_tensor = jnp.array(
        [
            [c + (-1) ** (c > 0) for c in clause]
            + [num_clauses] * (max_clause_len - len(clause))
            for clause in cnf_problem.clauses
        ]
    )
    var_tensor = jnp.array(
        [
            [abs(c) - 1 for c in clause]
            + [num_clauses] * (max_clause_len - len(clause))
            for clause in cnf_problem.clauses
        ]
    )
    sign_tensor = jnp.array(
        [
            [1 if c > 0 else -1 for c in clause] + [0] * (max_clause_len - len(clause))
            for clause in cnf_problem.clauses
        ]
    )
    # var_tensor = jnp.array(
    #     [
    #         [ abs(c)-1 for c in clause] + [num_clauses] * (max_clause_len - len(clause))
    #         for clause in cnf_problem.clauses
    #     ]
    # )
    # sign_tensor = jnp.array(
    #     [
    #         [ 1 if c > 0 else -1 for c in clause] + [0] * (max_clause_len - len(clause))
    #         for clause in cnf_problem.clauses
    #     ]
    # )
    if optimizer_str == "adamw":
        optimizer = optax.adamw(learning_rate=learning_rate)
    elif optimizer_str == "adam":
        optimizer = optax.adam(learning_rate=learning_rate)
    elif optimizer_str == "sgd":
        optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
    else:
        raise NotImplementedError
    labels = jnp.ones((batch_size, len(cnf_problem.clauses)))
    return embedding, optimizer, literal_tensor, labels


def run_back_prop(
    params: optax.Params,
    optimizer: optax.GradientTransformation,
    literal_tensor: jnp.ndarray,
    num_steps: int,
    loss_fn_str: str = "sigmoid_binary_cross_entropy",
) -> optax.Params:
    """with activation function"""

    def get_solutions(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        # assignment = (params>0.5).astype(int)
        assignment = (jax.nn.sigmoid(params) > 0.5).astype(int)
        sat = jnp.take(assignment, jnp.abs(literal_tensor), fill_value=1, axis=1)
        sat = jnp.where(literal_tensor > 0, 1 - sat, sat)
        sat = jnp.all(jnp.any(sat > 0, axis=2), axis=1)
        return jnp.take(assignment, jnp.where(sat)[0], axis=0)

    @jax.jit
    def check_terminate(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        # assignment = (params>0.5).astype(int)
        assignment = (jax.nn.sigmoid(params) > 0.5).astype(int)
        sat = jnp.take(assignment, jnp.abs(literal_tensor), fill_value=1, axis=1)
        sat = jnp.where(literal_tensor > 0, 1 - sat, sat)
        return jnp.any(jnp.all(jnp.any(sat > 0, axis=2), axis=1), axis=0)

    @jax.jit
    def compute_loss_l2(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        params = jax.nn.sigmoid(params)
        x = jnp.take(params, jnp.abs(literal_tensor), fill_value=1.0, axis=1)
        x = jnp.where(literal_tensor > 0, x, 1 - x)
        x = 1 - jnp.prod(x, axis=-1)
        labels = jnp.ones((x.shape[0], x.shape[1]))
        return optax.l2_loss(x, labels).mean()

    @jax.jit
    def step_l2(
        params: jnp.ndarray,
        opt_state: optax.OptState,
        literal_tensor: jnp.ndarray,
    ):
        loss_value, grads = jax.value_and_grad(compute_loss_l2)(params, literal_tensor)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # (assignment_func(params)>0.5).astype(int)
    @jax.jit
    def compute_loss(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        params = jax.nn.sigmoid(params)
        x = jnp.take(params, jnp.abs(literal_tensor), fill_value=1.0, axis=1)
        x = jnp.where(literal_tensor > 0, x, 1 - x)
        x = 1 - jnp.prod(x, axis=-1)
        labels = jnp.ones((x.shape[0], x.shape[1]))
        return optax.sigmoid_binary_cross_entropy(x, labels).mean()

    @jax.jit
    def step(
        params: jnp.ndarray,
        opt_state: optax.OptState,
        literal_tensor: jnp.ndarray,
    ):
        loss_value, grads = jax.value_and_grad(compute_loss)(params, literal_tensor)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # params = jax.nn.sigmoid(params)
    if loss_fn_str == "sigmoid_binary_cross_entropy":
        start_t = time.time()
        opt_state = optimizer.init(params)
        for step in range(num_steps):
            params, opt_state, loss_value = step(
                params=params,
                opt_state=opt_state,
                literal_tensor=literal_tensor,
            )
            is_complete = check_terminate(params, literal_tensor)
            if is_complete:
                break
        end_t = time.time()
    else:
        start_t = time.time()
        opt_state = optimizer.init(params)
        for step in range(num_steps):
            params, opt_state, loss_value = step_l2(
                params=params,
                opt_state=opt_state,
                literal_tensor=literal_tensor,
            )
            is_complete = check_terminate(params, literal_tensor)
            if is_complete:
                break
        end_t = time.time()
    solutions = get_solutions(params, literal_tensor)
    return params, step + 1, loss_value, end_t - start_t, solutions


def run_back_prop_verbose(
    params: optax.Params,
    optimizer: optax.GradientTransformation,
    literal_tensor: jnp.ndarray,
    num_steps: int,
    loss_fn_str: str = "sigmoid_binary_cross_entropy",
) -> optax.Params:
    def get_solutions(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        # assignment = (params>0.5).astype(int)
        assignment = (jax.nn.sigmoid(params) > 0.5).astype(int)
        sat = jnp.take(assignment, jnp.abs(literal_tensor), fill_value=1, axis=1)
        sat = jnp.where(literal_tensor > 0, 1 - sat, sat)
        sat = jnp.all(jnp.any(sat > 0, axis=2), axis=1)
        return jnp.take(assignment, jnp.where(sat)[0], axis=0)

    def check_terminate(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        # assignment = (params>0.5).astype(int)
        assignment = (jax.nn.sigmoid(params) > 0.5).astype(int)
        sat = jnp.take(assignment, jnp.abs(literal_tensor), fill_value=1, axis=1)
        sat = jnp.where(literal_tensor > 0, 1 - sat, sat)
        return jnp.any(jnp.all(jnp.any(sat > 0, axis=2), axis=1), axis=0)

    def compute_loss_l2(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        params = jax.nn.sigmoid(params)
        x = jnp.take(params, jnp.abs(literal_tensor), fill_value=1.0, axis=1)
        x = jnp.where(literal_tensor > 0, x, 1 - x)
        x = 1 - jnp.prod(x, axis=-1)
        labels = jnp.ones((x.shape[0], x.shape[1]))
        return optax.l2_loss(x, labels).mean()

    def step_l2(
        params: jnp.ndarray,
        opt_state: optax.OptState,
        literal_tensor: jnp.ndarray,
    ):
        l = compute_loss_l2(params, literal_tensor)
        loss_value, grads = jax.value_and_grad(compute_loss_l2)(params, literal_tensor)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # (assignment_func(params)>0.5).astype(int)

    def compute_loss(
        params: jnp.ndarray,
        literal_tensor: jnp.ndarray,
    ):
        params = jax.nn.sigmoid(params)
        x = jnp.take(params, jnp.abs(literal_tensor), fill_value=1.0, axis=1)
        x = jnp.where(literal_tensor > 0, x, 1 - x)
        x = 1 - jnp.prod(x, axis=-1)
        labels = jnp.ones((x.shape[0], x.shape[1]))
        return optax.sigmoid_binary_cross_entropy(x, labels).mean()

    def step(
        params: jnp.ndarray,
        opt_state: optax.OptState,
        literal_tensor: jnp.ndarray,
    ):
        loss_value, grads = jax.value_and_grad(compute_loss)(params, literal_tensor)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # params = jax.nn.sigmoid(params)
    if loss_fn_str == "sigmoid_binary_cross_entropy":
        start_t = time.time()
        opt_state = optimizer.init(params)
        for step in tqdm(range(num_steps), desc="Gradient Descent"):
            params, opt_state, loss_value = step(
                params=params,
                opt_state=opt_state,
                literal_tensor=literal_tensor,
            )
            is_complete = check_terminate(params, literal_tensor)
            if is_complete:
                break
        end_t = time.time()
    else:
        start_t = time.time()
        opt_state = optimizer.init(params)
        for step in tqdm(range(num_steps), desc="Gradient Descent"):
            params, opt_state, loss_value = step_l2(
                params=params,
                opt_state=opt_state,
                literal_tensor=literal_tensor,
            )
            is_complete = check_terminate(params, literal_tensor)
            if is_complete:
                break
        end_t = time.time()
    import pdb

    pdb.set_trace()
    solutions = get_solutions(params, literal_tensor)
    return params, step + 1, loss_value, end_t - start_t, solutions


# def run_back_prop_verbose(
#     params: optax.Params,
#     optimizer: optax.GradientTransformation,
#     var_tensor: jnp.ndarray,
#     sign_tensor: jnp.ndarray,
#     num_steps: int,
#     labels: jnp.ndarray,
#     loss_fn_str: str = "l2_loss",
# ) -> optax.Params:

#     def get_solutions(
#         params: jnp.ndarray,
#         var_tensor: jnp.ndarray,
#         sign_tensor: jnp.ndarray,
#     ):
#         # assignment = (params>0.5).astype(int)
#         assignment = (jax.nn.sigmoid(params) > 0.5).astype(int)
#         sat = jnp.take(assignment, var_tensor, fill_value=-1, axis=1)
#         sat = jnp.where(sign_tensor > 0, 1 - sat, sat)
#         sat = jnp.all(jnp.any(sat>0, axis=2), axis=1)
#         return jnp.take(assignment, jnp.where(sat)[0], axis=0)

#     def check_terminate(
#         params: jnp.ndarray,
#         var_tensor: jnp.ndarray,
#         sign_tensor: jnp.ndarray,
#     ):
#         # assignment = (params>0.5).astype(int)
#         assignment = (jax.nn.sigmoid(params) > 0.5).astype(int)
#         sat = jnp.take(assignment, var_tensor, fill_value=-1, axis=1)
#         sat = jnp.where(sign_tensor > 0, 1 - sat, sat)
#         return jnp.any(jnp.all(jnp.any(sat > 0, axis=2), axis=1), axis=0)

#     def compute_loss_l2(
#         params: jnp.ndarray, var_tensor: jnp.ndarray, sign_tensor: jnp.ndarray, labels: jnp.ndarray,
#     ):
#         params = jax.nn.sigmoid(params)
#         x = jnp.take(params, var_tensor, fill_value=0.0, axis=1)
#         x = jnp.where(sign_tensor > 0, x, 1 - x)
#         x = 1 - jnp.prod(x, axis=-1)
#         return optax.l2_loss(x, labels).mean()

#     def step_l2(
#         params: jnp.ndarray,
#         opt_state: optax.OptState,
#         var_tensor: jnp.ndarray,
#         sign_tensor: jnp.ndarray,
#         labels: jnp.ndarray,
#     ):
#         loss_value, grads = jax.value_and_grad(compute_loss_l2)(
#             params, var_tensor, sign_tensor, labels
#         )
#         updates, opt_state = optimizer.update(grads, opt_state)
#         params = optax.apply_updates(params, updates)
#         return params, opt_state, loss_value
# # (assignment_func(params)>0.5).astype(int)

#     def compute_loss(
#         params: jnp.ndarray, var_tensor: jnp.ndarray, sign_tensor: jnp.ndarray, labels: jnp.ndarray,
#     ):
#         params = jax.nn.sigmoid(params)
#         x = jnp.take(params, var_tensor, fill_value=0.0, axis=1)
#         x = jnp.where(sign_tensor > 0, x, 1 - x)
#         x = 1 - jnp.prod(x, axis=-1)
#         return optax.sigmoid_binary_cross_entropy(x, labels).mean()

#     def step(
#         params: jnp.ndarray,
#         opt_state: optax.OptState,
#         var_tensor: jnp.ndarray,
#         sign_tensor: jnp.ndarray,
#         labels: jnp.ndarray,
#     ):
#         loss_value, grads = jax.value_and_grad(compute_loss)(
#             params, var_tensor, sign_tensor, labels
#         )
#         updates, opt_state = optimizer.update(grads, opt_state)
#         params = optax.apply_updates(params, updates)
#         return params, opt_state, loss_value

#     # params = jax.nn.sigmoid(params)
#     if loss_fn_str == "sigmoid_binary_cross_entropy":
#         start_t = time.time()
#         opt_state = optimizer.init(params)
#         for step in tqdm(range(num_steps), desc="Gradient Descent"):
#             params, opt_state, loss_value = step(
#                 params=params,
#                 opt_state=opt_state,
#                 var_tensor=var_tensor,
#                 sign_tensor=sign_tensor,
#                 labels=labels,
#             )
#             is_complete = check_terminate(params, var_tensor, sign_tensor)
#             if is_complete:
#                 break
#         end_t = time.time()
#     else:
#         start_t = time.time()
#         opt_state = optimizer.init(params)
#         for step in tqdm(range(num_steps), desc="Gradient Descent"):
#             params, opt_state, loss_value = step_l2(
#                 params=params,
#                 opt_state=opt_state,
#                 var_tensor=var_tensor,
#                 sign_tensor=sign_tensor,
#                 labels=labels,
#             )
#             print(jax.nn.sigmoid(params[0]))
#             is_complete = check_terminate(params, var_tensor, sign_tensor)
#             if is_complete:
#                 break
#         end_t = time.time()
#     solutions = get_solutions(params, var_tensor, sign_tensor)
#     import pdb; pdb.set_trace()
#     return params, step+1, loss_value, end_t - start_t, solutions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    circuit = SATCircuit(
        use_pgates=True,
        key=jax.random.PRNGKey(0),
        batch_size=1000,
    )
    params, optimizer, var_tensor, sign_tensor, labels = circuit.init_problem(
        cnf_problem=CNF(
            from_file="/rscratch/jmk/projects/hwv/data/pigeon_hole_hard/pigeon_hole_10-SAT.cnf"
        ),
    )
    final_params, step, loss_value, elapsed_time = circuit.fit(
        num_steps=1000,
        params=params,
        optimizer=optimizer,
        var_tensor=var_tensor,
        sign_tensor=sign_tensor,
        labels=labels,
    )
    print(f"Time taken: {elapsed_time}")
    print(f"step {step}, loss: {loss_value}")
    # sat = circuit.check_sat(final_params, var_tensor, sign_tensor)
    # print(sat)
