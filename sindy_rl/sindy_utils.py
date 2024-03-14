import numpy as np
import pysindy as ps

def lin_and_cube_library(poly_int = False):
    '''
    Restricted polynomial library to cubic and linear combinations.
    
    Parameters:
        `poly_int` (bool): whether to into include polynomial (cubic)
            interaction terms (e.g. x^2 y). If False, just include homogenous
            terms (e.g. x^3, y^3)
    '''
    if poly_int:
        library_functions = [
            lambda x: x,
            lambda x,y: x * y**2,
            lambda x: x**3
        ]
        library_names = [lambda x: x, 
                         lambda x,y: f'{x} {y}^2', 
                         lambda x: f'{x}^3'
                         ]
    else:
        library_functions = [
            lambda x: x,
            lambda x: x**3
        ]
        library_names = [lambda x: x, lambda x: f'{x}^3']

    polyLibCustom = ps.CustomLibrary(library_functions=library_functions, 
                                    function_names = library_names)

    return polyLibCustom

def get_affine_lib(poly_deg, n_state=2, n_control = 2, poly_int=False, tensor=False, use_cub_lin=False):
    '''
    Create control-affine library of the form:
        x_dot = p_1(x) + p_2(x)u
    Where p_1 and p_2 are polynomials of degree poly_deg in state-variables, x. 
    
    Paramters:
        `poly_deg`  (int):  The degree of the polynomials p_1, p_2
        `n_state`   (int):  The dimension of the state variable x
        `n_control` (int):  The dimension of the control variable u
        `poly_int`  (bool): Whether to include the polynomial interactions
                    i.e., for poly_deg = 2, whether x_1 * x_2 terms will be used
                    or just (x_j)^2
        `tensor`    (bool): Whether or not to include p_2 (i.e the tensor product
                    of the polynomial state-space library and the linear control
                    library)
    
    '''
    if use_cub_lin:
        assert poly_deg==3, 'poly_deg must be 3 to use custom Cubic + Linear library'
        polyLib = lin_and_cube_library(poly_int = poly_int) 
    else:
        polyLib = ps.PolynomialLibrary(degree=poly_deg, 
                                        include_bias=False, 
                                        include_interaction=poly_int)
    affineLib = ps.PolynomialLibrary(degree=1, 
                                    include_bias=False, 
                                    include_interaction=False)

    # For the first library, don't use any of the control variables.
    # forcing this to be zero just ensures that we're using the "zero-th"
    # indexed variable. The source code uses a `np.unique()` call
    inputs_state_library = np.arange(n_state + n_control)
    inputs_state_library[-n_control:] = 0
    
    # For the second library, we only want the control variables.
    #  forching the first `n_state` terms to be n_state + n_control - 1 is 
    #  just ensuring that we're using the last indexed variable (i.e. the
    #  last control term).
    inputs_control_library = np.arange(n_state + n_control)
    inputs_control_library[:n_state] = n_state + n_control -1
    
    inputs_per_library = np.array([
                                    inputs_state_library,
                                    inputs_control_library
                                    ], dtype=int)

    if tensor:
        tensor_array = np.array([[1, 1]])
    else:
        tensor_array = None 

    generalized_library = ps.GeneralizedLibrary(
        [polyLib, affineLib],
        tensor_array=tensor_array,
        inputs_per_library=inputs_per_library,
    )
    return generalized_library

def get_affine_lib_from_base(base_lib, n_state=2, n_control = 2, include_bias = False):
    '''
    Returns library for regression of the form:
        W Phi(x) + W2 Phi(x) u
        
        `base_lib`  (ps.FeatureLibrary)
            base library, Phi
        `n_state`   (int):  The dimension of the state variable x
        `n_control` (int):  The dimension of the control variable u
    '''
    control_lib = ps.PolynomialLibrary(degree=1, 
                                    include_bias=False, 
                                    include_interaction=False)

    # For the first library, don't use any of the control variables.
    # forcing this to be zero just ensures that we're using the "zero-th"
    # indexed variable. The source code uses a `np.unique()` call
    inputs_state_library = np.arange(n_state + n_control)
    inputs_state_library[-n_control:] = 0
    
    # For the second library, we only want the control variables.
    #  forching the first `n_state` terms to be n_state + n_control - 1 is 
    #  just ensuring that we're using the last indexed variable (i.e. the
    #  last control term).
    inputs_control_library = np.arange(n_state + n_control)
    inputs_control_library[:n_state] = n_state + n_control -1
    
    inputs_per_library = np.array([
                                    inputs_state_library,
                                    inputs_control_library
                                    ], dtype=int)

    tensor_array = np.array([[1, 1]])

    libs = [base_lib, control_lib]
    if include_bias:
        libs = [ps.PolynomialLibrary(degree=0), base_lib, control_lib]
        tensor_array = np.array([[0, 1, 1,]])
        inputs_per_library = np.array([
                                    np.zeros(n_state + n_control),
                                    inputs_state_library,
                                    inputs_control_library,
                                    ], dtype=int)
    generalized_library = ps.GeneralizedLibrary(
        libs,
        tensor_array=tensor_array,
        inputs_per_library=inputs_per_library,
    )
    return generalized_library



def build_optimizer(config):
    '''
    Helper method to build a SINDy optimizer from a configuration dictionary
    
    Example Structure:
    'base_optimizer':
        'name': 'STLSQ'
        'kwargs': 
            'alpha': 1e-5
            'thresh': 1e-2
    'ensemble':
        'bagging': true
        'library_ensemble': true
        'n_models': 100
    '''
    opt_name = config['base_optimizer']['name']
    opt_kwargs = config['base_optimizer']['kwargs']
    base_optimizer = getattr(ps, opt_name)(**opt_kwargs)
    
    ensemble = config['ensemble']
    if ensemble:
        optimizer = ps.EnsembleOptimizer(opt=base_optimizer,
                                         **ensemble)
    else:
        optimizer =base_optimizer

    return optimizer

def build_feature_library(config):
    '''Build library from config'''
    # TO-DO: Make more general
    lib_name = config['name']
    lib_kwargs = config['kwargs']
    
    # custom affine library
    if lib_name == 'affine':
        lib = get_affine_lib(**lib_kwargs)
    else:
        lib_class = getattr(ps, lib_name)
        lib = lib_class(**lib_kwargs)
    return lib

#---------------------------------------------------
# Some spare code for taking strings and then building
# library functions from them.
#---------------------------------------------------
# import sympy
# from pysindy import CustomLibrary
# lib_names = ['{}', '{}^3', 'sin(2*{})', 'exp({})']
# lib_name_fns = [sym.format for sym in lib_names]

# var = sympy.var('x')
# lib_syms = [sympy.sympify(lib_fn('x')) for lib_fn in lib_name_fns]
# tmp_fn = [sympy.lambdify(var, sym, 'numpy') for sym in lib_syms]
# tmp = CustomLibrary(library_functions=tmp_fn, function_names=lib_name_fns)
# tmp.fit(np.ones(2))
# tmp.get_feature_names(['x', 'y'])


if __name__ == '__main__': 
    lib = get_affine_lib(poly_deg=2)
    