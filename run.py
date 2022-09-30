import jax.numpy as jnp
from jax import grad, jit, vmap, tree_map
from jax import random
from jax import nn
import timeit


def linear_regression(w, x):
    return jnp.dot(w, x)

def relu(x):
    return jnp.where(x > 0, x, 0)

def layernorm(x):
    m = jnp.mean(x)
    x -= m
    v = jnp.mean(jnp.square(x))
    x = x / jnp.sqrt(v + 1e-8)
    return x

def recursive_mlp(w, x):
    for i in range(1):
        x = layernorm(nn.relu(jnp.dot(w, x)))
    x = jnp.dot(w, x)
    return x

def recursive_mlp_regression(w, x):
    return jnp.sum(recursive_mlp(w, x))

def mlp(w, x):
    w1, w2, w3 = w
    x = nn.relu(jnp.dot(w1, x))
    x = nn.relu(jnp.dot(w2, x))
    x = jnp.dot(w3, x)
    return x

def mlp_regression(w, x):
    return jnp.sum(mlp(w, x))


N = 10
d = 2
n1 = 10
n2 = 10
n3 = 10
key = random.PRNGKey(0)
key, subkey = random.split(key)
# w = random.normal(subkey, (1, d), dtype=jnp.float32)
# w = jnp.ones((1, d), dtype=jnp.float32)
w1 = random.normal(subkey, (n1, d), dtype=jnp.float32)
key, subkey = random.split(key)
w2 = random.normal(subkey, (n2, n1), dtype=jnp.float32)
key, subkey = random.split(key)
w3 = random.normal(subkey, (n3, n2), dtype=jnp.float32)
w = w1, w2, w3
key, subkey = random.split(key)
x = random.normal(subkey, (N, d), dtype=jnp.float32)
key, subkey = random.split(key)
y = jnp.mean(x, axis=1, keepdims=True)

batched_recursive_mlp_regression = vmap(recursive_mlp_regression, in_axes=(None, 0), out_axes=0)
batched_mlp_regression = vmap(mlp_regression, in_axes=((None, None, None), 0), out_axes=0)
batched_linear_regression = vmap(linear_regression, in_axes=(None, 0), out_axes=0)

@jit
def loss(w, x, y):
    y_hat = batched_mlp_regression(w, x)
    return jnp.mean(jnp.square(y - y_hat))


grad_loss = grad(loss)

@jit
def update_weight(w, x, y):
    g = grad_loss(w, x, y)
    lr = .001
    return tree_map(lambda w, g: w - lr * g, w, g)

print(loss(w, x, y))
for i in range(200):
    w = update_weight(w, x, y)
    print(loss(w, x, y))
# print(w)
