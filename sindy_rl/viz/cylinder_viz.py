import numpy as np
import firedrake as fd 
import matplotlib.pyplot as plt
import seaborn as sns
import os

from hydrogym import firedrake as hgym

from sindy_rl.hydroenv import CylinderLiftEnv


def grab_env(checkpath, Re=100, dt = 1e-2, mesh = 'medium'):
    hydro_config = {
        "flow_config": {
            'actuator_integration': 'implicit',
            'mesh': mesh,
            'Re': Re,
            'restart': checkpath
        },
        "solver": hgym.IPCS,
        "solver_config":
            {
                'dt': dt
            }
    }

    env_config = {
            'hydro_config': hydro_config,
            'control_freq': 10,   
            'n_skip_on_reset': 0, 
            'max_episode_steps': 1000000
    }
    env = CylinderLiftEnv(env_config)
    return env

def basic_plot(env, square = False, figsize = (15,6)):
    
    fig, ax = plt.subplots(figsize=figsize)
    clim = (-2, 2)
    levels = np.linspace(*clim, 16)

    vort = fd.project(fd.curl(env.flow.u), env.flow.pressure_space)

    im = fd.tripcolor(vort, 
                      cmap = sns.color_palette("icefire", as_cmap=True),
                      axes=ax, vmin = clim[0], vmax = clim[1])
    cyl = plt.Circle((0, 0), 0.5, edgecolor="k", facecolor="gray")
    im.axes.add_artist(cyl)
    ax.set_facecolor('k')
    ax.set_xlim([-2, 9])
    ax.set_ylim([-2, 2])
    ax.set_xticks([])
    ax.set_yticks([])

    if square:
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        
    return fig, ax
    
    
def cylinderMultiPlot(fig, ax, env, vort_lim = [-2,2], p_lim = [-1,1], vel_lim = [-1,1]):
    clim = vort_lim
    vort = fd.project(fd.curl(env.flow.u), env.flow.pressure_space)
    im = fd.tripcolor(vort,
                      cmap = sns.color_palette("icefire", as_cmap=True),
                      axes=ax[0,0], vmin = clim[0], vmax = clim[1])
    ax[0,0].set_xlim(-0.6, 0.9)
    ax[0,0].set_ylim(-0.75, 0.75)
    
    clim = p_lim
    im_p = fd.tripcolor(env.flow.p,
        cmap = sns.color_palette("icefire", as_cmap=True),
        axes=ax[0,1], vmin = clim[0], vmax = clim[1]) 
    ax[0,1].set_xlim(-0.6, 0.9)
    ax[0,1].set_ylim(-0.75, 0.75)
    U = fd.assemble(fd.project(env.flow.u[0], env.flow.pressure_space))
    V = fd.assemble(fd.project(env.flow.u[1], env.flow.pressure_space))
    
    clim = vel_lim
    im_U = fd.tripcolor(U,
        cmap = sns.color_palette("icefire", as_cmap=True),
        axes=ax[1,0], vmin = clim[0], vmax = clim[1])
    im_V = fd.tripcolor(V,
        cmap = sns.color_palette("icefire", as_cmap=True),
        axes=ax[1,1], vmin = clim[0], vmax = clim[1])
    ax[1,0].set_xlim(-0.6, 0.9)
    ax[1,0].set_ylim(-0.75, 0.75)
    ax[1,1].set_xlim(-0.6, 0.9)
    ax[1,1].set_ylim(-0.75, 0.75)
    ax[0,0].set_title('Vorticity')
    ax[0,1].set_title('Pressure')
    ax[1,0].set_title('X Velocity')
    ax[1,1].set_title('Y Velocity')
    return fig, ax

if __name__ == '__main__': 
    import pickle
    from tqdm import tqdm
    import matplotlib
    matplotlib.use('Agg')
    
    from sindy_rl import _parent_dir
    
    eval_dir = os.path.join(_parent_dir, 'data', 'hydro', 'cylinder', 'eval', 
                            '2023-12-22_medium',
                            'dt=1e-2_agent=13_check=200'
                            )
    
    _DT = 1e-1
    save_type = 'flow'
    steps = np.arange(900)
    SAVE_OBS = False
    
    if save_type == 'flow': 
        save_dir = os.path.join(eval_dir, 'flow_images')
    elif save_type == 'multi': 
        save_dir = os.path.join(eval_dir, 'multi-plot_images')
    else:
        raise ValueError('invalid save_type')

    os.makedirs(save_dir, exist_ok=True)
    print(save_type, save_dir)
    
    real_obs = []
    for step_idx in tqdm(steps):
        checkpath = os.path.join(eval_dir, f'flow_{step_idx:06d}.ckpt')
        save_path = os.path.join(save_dir, f'flow_{step_idx:06d}.png')
        
        if os.path.exists(save_path):
            continue

        env = grab_env(checkpath)
        if SAVE_OBS:
            real_obs.append(env.flow.get_observations())
        
        t = _DT * step_idx
        
        if save_type =='flow': 
            fig, ax = basic_plot(env, square=False)
            ax.set_title(f't = {t:.2f}', fontsize=20)
        elif save_type == 'multi':
            fig, axes = plt.subplots(2,2,figsize=(10,10))
            fig, axes = cylinderMultiPlot(fig, axes, env)
            fig.suptitle(f't = {t:.2f}', fontsize=20)
        else:
            raise ValueError('invalid save_type')

        plt.savefig(save_path)
        plt.close()
        env.close()
        
    if SAVE_OBS:
        real_obs = np.array(real_obs)
        fname = os.path.join(eval_dir, 'checkpoints_raw_obs.pkl')
        
        with open(fname, 'rb') as f:
            pickle.dump(real_obs)