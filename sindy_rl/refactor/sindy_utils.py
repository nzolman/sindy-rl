import pysindy as ps

from sindy_rl.sindy_utils import get_affine_lib_from_base, get_affine_lib


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