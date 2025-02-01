# ----------------------------------------------------
# Used for generating steady flow checkpoints
# Compatibale with the updated version of Hydrogym
# ----------------------------------------------------


import hydrogym.firedrake as hgym
import firedrake as fd
import os

from sindy_rl import _parent_dir, _data_dir

output_dir = os.path.join(_data_dir, 'steady_checkpoints')

def ramp_step(flow, Re_init, save_name, save_all=True):
    '''
    Iteratively solve for the steady state at increasing Re.
    
    flow: (hgym.FlowConfig)
        flow to solve for
    Re_init: (list)
        Re values (in increasing order)
    save_name: (str) 
        path of string to save at.
    save_all: (bool)
        whether to save all intermediate checkpoints or just the last. 
    '''
    solver_parameters = {"snes_monitor": None}
    for i, Re in enumerate(Re_init):
        # Since this flow is at high Reynolds number we have to
        #    ramp to get the steady state
        flow.Re.assign(Re)
        hgym.print(f"Steady solve at Re={Re_init[i]}")
        solver = hgym.NewtonSolver(flow, solver_parameters=solver_parameters)
        qB = solver.solve()
        
        check_path = save_name + f'_Re={Re}.ckpt'
        if save_all:
            flow.save_checkpoint(check_path)
    
    flow.save_checkpoint(check_path)
    return flow

def gen_steady(flow_name, flow_config, Re_init, save_all=True):
    '''
    Iteratively solve for the steady state at increasing Re.
    
    flow_name: (str)
        flow to solve for
    flow_config: (dict)
        configuration dictionary for flow
    Re_init: (list)
        Re values (in increasing order)
    save_all: (bool)
        whether to save all intermediate checkpoints or just the last. 
    '''
    
    print(flow_name)
    mesh = flow_config.get('mesh', '')
    flow = getattr(hgym, flow_name)(Re=Re_init[0], **flow_config)
    
    dof = flow.mixed_space.dim()
    hgym.print(f"Total dof: {dof} --- dof/rank: {int(dof/fd.COMM_WORLD.size)}")

    save_name = os.path.join(output_dir, f'{flow_name}-{mesh}-steady')
    
    flow = ramp_step(flow, Re_init, save_name, save_all=save_all)
    
    return flow

if __name__ == '__main__': 
    flow_config = {'mesh': 'medium'}
    
    # gen_steady('Cylinder', flow_config, Re_init=[30, 100, 150, 200, 250])

    gen_steady('Pinball', flow_config, Re_init=[30, 100, 130, 150, 175, 200, 225, 250, 275, 300, 325, 350])