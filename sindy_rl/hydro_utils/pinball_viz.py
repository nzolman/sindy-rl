import numpy as np
import os
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt

import firedrake as fd
import seaborn as sns
from hydrogym import firedrake as hgym
from sindy_rl.hydroenv import PinballLiftEnv

def grab_env(checkpath, flow='track'):
    hydro_config = {
        "flow": 'square',
        "flow_config": {
            'actuator_integration': 'implicit',
            'mesh':'fine',
            'Re': 30,
            'restart': checkpath
        },
        "solver": hgym.IPCS,
        "solver_config":
            {
                'dt': 1e-2
            }
    }

    env_config = {
            'hydro_config': hydro_config,
            'control_freq': 10,   
            'n_skip_on_reset': 0, 
            'max_episode_steps': 1000000
    }
    env = PinballLiftEnv(env_config)
    return env



def basic_plot(env, square = False, plt_circles=False):
    if square: 
        figsize=(10,10)
    else:
        figsize = (14,8)
    fig, ax = plt.subplots(figsize=figsize)
    clim = (-2, 2)
    levels = np.linspace(*clim, 16)

    vort = fd.project(fd.curl(env.flow.u), env.flow.pressure_space)

    im = fd.tripcolor(vort, 
                      cmap = sns.color_palette("icefire", as_cmap=True), # vlag
                      axes=ax, vmin = clim[0], vmax = clim[1])
    
    if plt_circles:
        cyl1 = plt.Circle((0, 0), 0.5, edgecolor="k", facecolor="gray")
        im.axes.add_artist(cyl1)

        cyl2 = plt.Circle((1.3995, 0.75), 0.5, edgecolor="k", facecolor="gray")
        im.axes.add_artist(cyl2)

        cyl3 = plt.Circle((1.3995, -0.75), 0.5, edgecolor="k", facecolor="gray")
        im.axes.add_artist(cyl3)

        ax.set_facecolor('k')
    ax.set_xlim([-2, 12])
    ax.set_ylim([-4, 4])
    ax.set_xticks([])
    ax.set_yticks([])

    if square:
        ax.set_xlim(-1,3)
        ax.set_ylim(-2,2)
        
    return fig, ax
    
    
def pinballMultiPlot(fig, ax, env):
    xlim = (-1,3)
    ylim = (-2,2)
    
    clim = [-2,2]
    vort = fd.project(fd.curl(env.flow.u), env.flow.pressure_space)
    im = fd.tripcolor(vort,
                      cmap = sns.color_palette("icefire", as_cmap=True), # vlag
                      axes=ax[0,0], vmin = clim[0], vmax = clim[1])

    clim = [-1,1]
    im_p = fd.tripcolor(env.flow.p,
        cmap = sns.color_palette("icefire", as_cmap=True),
        axes=ax[0,1], vmin = clim[0], vmax = clim[1]) 

    U = fd.assemble(fd.project(env.flow.u[0], env.flow.pressure_space))
    V = fd.assemble(fd.project(env.flow.u[1], env.flow.pressure_space))
    
    im_U = fd.tripcolor(U,
        cmap = sns.color_palette("icefire", as_cmap=True),
        axes=ax[1,0], vmin = clim[0], vmax = clim[1])
    
    im_V = fd.tripcolor(V,
        cmap = sns.color_palette("icefire", as_cmap=True),
        axes=ax[1,1], vmin = clim[0], vmax = clim[1])
    
    for a in ax.flatten():
        a.set_xlim(*xlim)
        a.set_ylim(*ylim)

    ax[0,0].set_title('Vorticity')
    ax[0,1].set_title('Pressure')
    ax[1,0].set_title('X Velocity')
    ax[1,1].set_title('Y Velocity')
    return fig, ax



if __name__ == '__main__':
    import matplotlib
    from sindy_rl import _parent_dir
    matplotlib.use('Agg')


    eval_dir = os.path.join(_parent_dir, 'data', 'hydro', 'pinball', 'eval', 
                            '2023-12-29_fine-classic',
                            'dt=1e-2_long'
                            )
    
    _DT = 1e-1
    save_type = 'multi'
    steps = np.arange(600)
    
    if save_type == 'flow': 
        save_dir = os.path.join(eval_dir, 'flow_images')
    elif save_type == 'multi': 
        save_dir = os.path.join(eval_dir, 'multi-plot_images')
    else:
        raise ValueError('invalid save_type')

    os.makedirs(save_dir, exist_ok=True)
    print(save_type, save_dir)
    
    for step_idx in tqdm(steps):
        checkpath = os.path.join(eval_dir, f'flow_{step_idx:06d}.ckpt')
        save_path = os.path.join(save_dir, f'flow_{step_idx:06d}.png')
        
        if os.path.exists(save_path):
            continue

        env = grab_env(checkpath)
        t = _DT * step_idx
        
        if save_type =='flow': 
            fig, ax = basic_plot(env, square=False)
            ax.set_title(f't = {t:.2f}', fontsize=20)
        elif save_type == 'multi':
            fig, axes = plt.subplots(2,2,figsize=(10,10))
            fig, axes = pinballMultiPlot(fig, axes, env)
            fig.suptitle(f't = {t:.2f}', fontsize=20)
        else:
            raise ValueError('invalid save_type')

        plt.savefig(save_path)
        plt.close()
        env.close()
        