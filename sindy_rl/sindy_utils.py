import numpy as np
import pysindy as ps

def get_affine_lib(poly_deg, n_state=2, n_control = 2, poly_int=False, tensor=False):
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

if __name__ == '__main__': 
    lib = get_affine_lib(poly_deg=2)
    