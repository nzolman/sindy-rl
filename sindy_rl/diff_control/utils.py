from abc import ABC, abstractmethod
from jax import numpy as jnp
from jax import jit, grad, value_and_grad
from jax.random import PRNGKey
from jax.experimental.ode import odeint
import jax
import optax
from functools import partial
import numpy as np


@jit
def mlp(params, inputs):
  '''A multi-layer perceptron, i.e. a fully-connected neural network.

    Inputs:
    - params: list of (weights, bias) pairs
    - inputs (jnp.array): Inputs into neural network

    Returns:
    - outputs: (jnp.array): Outputs of neural network
  '''
  for w, b in params:
    outputs = jnp.dot(inputs, w) + b  # Linear transform
    inputs = jnp.tanh(outputs)        # Nonlinearity
  return outputs


def init_params(key, initializer, layer_sizes):
    '''
    TO-DO: use dictionaries instead of tuples
    '''
    keys = jax.random.split(key, num = 2*len(layer_sizes[:-1]))
    initializer = initializer
    weights = [initializer(subkey, (m, n)) for (subkey, m, n) in zip(keys[::2], layer_sizes[:-1], layer_sizes[1:])]
    
    biases = [initializer(subkey, (n,)) for (subkey, m, n) in zip(keys[1::2], layer_sizes[:-1], layer_sizes[1:])]

    key = jax.random.split(keys[-1], num=1)[0]
    return key, list(zip(weights,biases))