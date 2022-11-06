from dataclasses import replace
import re
from sympy import sympify
from sympy2jax import SymbolicModule

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


def replace_expr(expression, feature_names):
    '''
    Replaces expression with multiplication symbols based on the feature names.
        For example, for the expression 'x0x0u0x0_t' with feature names ['x0', 'u0']
        this will become: 'x0*x0*u0*x0_t'
    '''
    tmp_string = expression
    
    # have to run this twice in order to adequately replace the strings
    #  do to how the replace function works. Otherwise, you might have
    #  u0u0u0
    #  which will first replace u0u0 to u0*u0, but it won't identify the second
    #  u0 as one that needs to be replaced.
    for _ in range(2):
        for feat1 in feature_names:
            for feat2 in feature_names:
                tmp_string = tmp_string.replace(f'{feat1}{feat2}', f'{feat1}*{feat2}')
    # tmp_string.replace(' ', '*')
    return tmp_string

def strToJaxFn(eq_strs):
    '''
    Convert set of strings to a JAX function.

    Argument:
        - `eq_strs` (list): list of strings that represent an expression.
            e.g. ['x0 + x1', 'x1 + x2']
    Returns:
        - `fn` (function): function that accepts keyword arguments 
            of the variables defined in the string expression. 
            e.g. fn(x0 = jnp.array([1.0, 2.0]), x1 = jnp.array([2.3, 7.2]))
    Notes:
        - the sizes of the variables being passed should be compatible with 
            the operations in the expressions. In practice, this means that
            they should either be the same size or scaler quantities. 

        - Convenient to define dictionary and then call it.
            e.g. 
                x_dict = {'x0': jnp.array([1.0, 2.0]), 
                          'x1': jnp.array([2.3, 7.2])
                        }
                fn(**x_dict)
    '''
    eq_syms = [sympify(eq_str) for eq_str in eq_strs]
    fn = SymbolicModule(eq_syms)
    return fn

def basic_formatter(eqs):
    '''
    Simply make the string replacements of the form 
        x0 -> *x0 and u0 -> *u0
        Enables multiplication of variables for sympy to handle.
    
    Arguments:
        `eqs` (list): list of equation strings.
    Returns:
        modified list of equation strings with the appropriate variables replaced.
    '''
    return [eq.replace('x', '*x').replace('u', '*u').replace(' 1 ', '*1') for eq in eqs]

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
    from pprint import pprint

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
    model.fit(x_train, u = np.ones(len(x_train)), t=dt, ensemble=False, quiet=True)

    # model.print()
    eqs = model.equations(precision=10)

    # hacky way to incorporate multiplication from spacings
    formatted_eqs = basic_formatter(eqs)
    
    fn = strToJaxFn(formatted_eqs)
    
    from jax import jit
    @jit
    def dynamics(x,t):
        x0, x1, x2 = x.T
        u0 = 1.0
        return fn(x0=x0, x1=x1, x2=x2, u0 = u0)
    

    sol = odeint(dynamics, jnp.array(x0_train, dtype=jnp.float32), t_train)
    print(sol)