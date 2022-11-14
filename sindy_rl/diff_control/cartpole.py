from jax import numpy as jnp
import numpy.random as npr
from jax import jit, grad, value_and_grad
from jax.experimental.ode import odeint
import optax
import time
from functools import partial

# from sindy_rl.diff_control.trainer import BaseDiffControlTrainer

# Code adapted from Julia [1,2] to JAX
# 
# [1] https://medium.com/swlh/neural-ode-for-reinforcement-learning-and-nonlinear-optimal-control-cartpole-problem-revisited-5408018b8d71
# [2] https://github.com/paulxshen/neural-ode-cartpole/blob/master/Cartpole.jl

# physical params
# TO-DO: figure out way to stuff these in dynamics function

m = 1 # pole mass kg
M = 2 # cart mass kg
L = 1 # pole length m
g = 9.8 # acceleration constant m/s^2

modpi = lambda theta: (theta + jnp.pi)%(2*jnp.pi) - jnp.pi

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


def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
    '''
    Initialize parameters for MLP
    '''
    return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

@jit
def controller(params, inputs):
    return mlp(params, inputs)


def cartpole(z,t, control =lambda z,t: 0):
    '''
    dynamics for cartpole with force as state
    
    This is a hacky way of approximately keeping track of the force being applied at each 
    step. The other way would be to pass propagated states back through the NN in post
    and calculate the force. 

    Inputs:
    - z: state
    - t: time
    - *params: NN parameters
    '''
    # position (cart), velocity, pole angle, angular velocity
    x, dx, theta, dtheta, _ = z

    force = control(z,t)
    
    dz = jnp.array([
                dx,
                (force + m * jnp.sin(theta) * (L * dtheta**2 - g * jnp.cos(theta))) / (M + m * jnp.sin(theta)**2),
                dtheta,
                (-force * jnp.cos(theta) - m * L * dtheta**2 * jnp.sin(theta) * jnp.cos(theta) + (M + m) * g * jnp.sin(theta)) / (L * (M + m * jnp.sin(theta)**2)),
                force
            ])
    return dz

def cartpole_basic(z,t):
    control = lambda z,t: 0
    return cartpole(z,t,control=control)

def cartpole_random(z,t, t_collect= jnp.arange(0,1,0.02)):
    forces=  npr.normal(loc=0, scale =1,size=t_collect.shape)
    control = lambda z, t: jnp.interp(t, t_collect, forces)
    return cartpole(z, t, control = control)

def cartpole_params(z,t, *params):
        # force = controller(params, state_in).T[0]
    def control(z,t):
        x, dx, theta, dtheta, _ = z
        state_in = jnp.array([jnp.cos(theta), jnp.sin(theta), dtheta]).T
        return controller(params, state_in).T[0]
    return cartpole(z,t, control=control)

@jit
def predict_neural_ode(params, z0, t):
    '''
    Predict forward

    Inputs:
    -params: NN parameters
    '''
    res = odeint(cartpole_params, z0 ,t , *(params))
    return res


@jit
def loss_neural_ode(params, z0, t):
    '''
    Determine loss

    Inputs:
    - params: NN paramaters
    '''
    pred = predict_neural_ode(params, z0, t)
    x, dx, theta, dtheta, impulse = pred.T
    #x, dx, theta, dtheta, impulse = jnp.transpose(pred, (1,0,2))
    # print( impulse.shape, jnp.diff(impulse,axis=0).shape,jnp.diff(t).shape)
    forces = jnp.diff(impulse)/jnp.diff(t)
    theta = modpi(theta)
    # loss = jnp.mean(theta**2) + 4*theta[-1]**2 + dx[-1]**2 
    #       + dtheta[-1]**2 +0.1* jnp.mean(x**2) + 0.001*jnp.max(forces**2)#+ dtheta[-1]**2
    loss = 100*jnp.mean(theta**2) + jnp.mean(dtheta**2) + jnp.mean(dx**2) + jnp.mean(x**2) + 0.0001*jnp.mean(forces**2) + 0.001*jnp.max(forces**2)
    # only have integral costs on everything except x
    # big penalty coeff on derivatives.
    return loss

@jit
def v_loss_neural_ode(params, z0, t):
    '''
    Determine loss

    Inputs:
    - params: NN paramaters
    '''
    pred = predict_neural_ode(params, z0, t)
    x, dx, theta, dtheta, impulse = jnp.transpose(pred, (1,0,2))

    dt = jnp.diff(t)[0]
    forces = jnp.diff(impulse,axis=0)/dt
    theta = modpi(theta)

    loss = 100*jnp.mean(theta**2) + jnp.mean(dtheta**2) + jnp.mean(dx**2) + jnp.mean(x**2) + 0.0001*jnp.mean(forces**2) #+ 0.001*jnp.max(forces**2)
    return loss



if __name__ == '__main__':
    from jax import vmap, pmap
    tic = time.time()
    # Hyperparameters.
    controller_layers = [3, 8, 1]

    param_scale = 1.0
    step_size = 1e-3
    train_iters = 1000
    z0 =jnp.array([0, 0, jnp.pi, 0, 0])
    t = jnp.linspace(0,1,100)

    controller_params = init_random_params(param_scale, controller_layers,rng=npr.RandomState(1))
    optimizer = optax.adam(learning_rate=step_size)
    opt_state = optimizer.init(controller_params)

    v_batch_loss = vmap(loss_neural_ode, in_axes= (None, 0, None))
    p_batch_loss = pmap(loss_neural_ode, in_axes= (None, 0, None)) # export XLA_FLAGS="--xla_force_host_platform_device_count=16"
    # batch_loss = jit(lambda params, z0, t: jnp.mean(v_batch_loss(params, z0, t)))
    # batch_loss = jit(lambda params, z0, t: jnp.mean(p_batch_loss(params, z0, t)))
    batch_loss = lambda params, z0, t: jnp.mean(p_batch_loss(params, z0, t))
    
    @jit
    def train_step(params, opt_state, z0, t):
        '''
        Apply one gradient update
        '''
        # loss, grads = value_and_grad(loss_neural_ode)(params, z0, t)
        loss, grads = value_and_grad(batch_loss)(params, z0, t)
        # loss, grads = value_and_grad(v_loss_neural_ode)(params, z0, t)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, opt_state
    


    for i in range(int(1e4)):
        # random init conditions might help? 
        # z0 =jnp.array([2*npr.random()-1, 
        #             2*npr.random()-1, 
        #             npr.random()-0.5 + jnp.pi, 
        #             2*npr.random()-1, 0])
        
        
        z0 = jnp.array([0, 0, jnp.pi, 0, 0])
        # loss, controller_params, opt_state = train_step(controller_params, opt_state, z0, t)
        z0 = jnp.array([z0, z0, z0, z0])
        loss, controller_params, opt_state = train_step(controller_params, opt_state, z0, t)
        if i % 1000 == 0:
            # z0 =jnp.array([0, 0, jnp.pi, 0, 0])
            z0 =jnp.array([[0, 0, jnp.pi, 0, 0]])
            loss, controller_params, opt_state =  train_step(controller_params,opt_state, z0, t)
            print(i, loss)
    toc = time.time()


    print(toc-tic)