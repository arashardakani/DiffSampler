import jax
import jax.numpy as jnp

def g(W, x):
  y = jnp.dot(W, x)
  return jnp.sin(y)

def f(W1, W2, W3, x):
  x = g(W1, x)
  x = g(W2, x)
  x = g(W3, x)
  return x

W1 = jnp.ones((5, 4))
W2 = jnp.ones((6, 5))
W3 = jnp.ones((7, 6))
x = jnp.ones(4)

# Inspect the 'residual' values to be saved on the forward pass
# if we were to evaluate `jax.grad(f)(W1, W2, W3, x)`
from jax.ad_checkpoint import print_saved_residuals
jax.ad_checkpoint.print_saved_residuals(f, W1, W2, W3, x)
