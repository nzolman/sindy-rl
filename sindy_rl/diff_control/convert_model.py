import re

def format_eq(eq):
  '''
  Convert SINDy String to 
  '''
  # works for continuous SINDy w/ polynomial dynamics + control
  rep = { ' x': '*x',  # x multiplication
          ' u': '*x',  # u multiplication
          '^': '**',   # exponentiation
          ' 1 ': '*1 ' # intercept mupltiplication
          }

  rep = dict((re.escape(k), v) for k,v in rep.items())
  pattern = re.compile('|'.join(rep.keys()))
  eq_text = pattern.sub(lambda m: rep[re.escape(m.group(0))], eq)

  return eq_text

def eq2dyn(x, eq=None, u=None):
    '''
    !!!!!!!!!!!
    **WARNING**
        DO NOT COPY AND PASTE UNLESS YOU UNDERSTAND WHAT
        IS GOING ON HERE. BAD PRACTICE TO USE 
        `eval` AND `exec`!!!
    !!!!!!!!!!!

    Yes, this is hacky. Yes, there's a better way.
    But this is cheap and it's compatible with the SINDy library.
    The variables are also only defined locally, so we don't have
    to worry about them spilling over. 
    '''
    # intialize variables x0, x1, ... with values from x
    for i in range(len(x)):
        exec(f'x{i} = x[{i}]')
    
    if type(u) != type(None):
        # initalize variables u0... with values from u
        for i in range(len(x)):
            exec(f'u{i} = u[{i}]')
    
    # format the string and evaluate the expression.
    eq_str = format_eq(eq)
    return eval(eq_str)


if __name__ == '__main__':
    '''
    Example to make sure this runs in practice!
    '''

    import jax.numpy as jnp
    from jax.experimental.ode import odeint
    from scipy.integrate import solve_ivp
    import numpy as np
    import pysindy as ps
    from pysindy.utils.odes import lorenz

    dt = .002
    t_train = np.arange(0, 10, dt)
    t_train_span = (t_train[0], t_train[-1])
    x0_train = [-8, 8, 27]
    x_train = solve_ivp(lorenz, t_train_span, x0_train, 
                        t_eval=t_train).y.T

    feature_names=None
    optimizer = ps.STLSQ()
    model = ps.SINDy(feature_names=feature_names, optimizer=optimizer)
    # Ensemble with replacement (V1)
    model.fit(x_train, t=dt, ensemble=False, quiet=True)
    eqs = model.equations(precision=10)

    from jax import jit
    @jit
    def dynamics(x, t):
        return jnp.array([eq2dyn(x, eq) for eq in eqs])

    sol = odeint(dynamics, jnp.array(x0_train, dtype=jnp.float32), t_train)
    print(sol)