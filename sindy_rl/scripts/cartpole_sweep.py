'''Used for generating experiments for Appendix C.1.1 "On-Policy Collection Study"'''

import warnings
warnings.filterwarnings('ignore')

import logging
import os
import numpy as np
from ray.rllib.algorithms.registry import ALGORITHMS as rllib_algos
from ray.tune.schedulers import PopulationBasedTraining
from ray import tune, air

from sindy_rl import _parent_dir
from sindy_rl.pbt_dyna import dyna_sindy, explore


_FREQ = 40
_PI_COLLECT = 1000
_OFF_BUFFER = 8000
_ON_BUFFER = 8000

def update_config(config):
    config['exp_dir'] = 'cart-swingup' # 'cart-swingup-sweep'
    config['dyn_fit_freq'] = _FREQ
    
    config['off_policy_buffer']['init']['kwargs']['n_steps'] = _OFF_BUFFER
    config['on_policy_buffer']['config']['max_samples'] = _ON_BUFFER
    config['on_policy_buffer']['collect']['n_steps'] = _PI_COLLECT
    config['ray_config']['run_config']['name'] = f'freq={int(_FREQ)}_coll={int(_PI_COLLECT)}_buff={_OFF_BUFFER}_{_ON_BUFFER}_ens-rew=fixed_ens-dyn=med-20_steps=1k'
    return config
    
if __name__ == '__main__': 
    import yaml
    import ray
    from pprint import pprint
    
    from sindy_rl.policy import RandomPolicy
    filename = os.path.join(_parent_dir, 
                            'sindy_rl',
                            'config_templates', 
                            'dyna_config_cart.yml')
    
    with open(filename, 'r') as f:
        dyna_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    dyna_config = update_config(dyna_config)
    pprint(dyna_config)

    LOCAL_DIR =  os.path.join(_parent_dir, 'ray_results', dyna_config['exp_dir'])
    
    # Setup logger
    logging.basicConfig()
    logger = logging.getLogger('dyna-sindy')
    logger.setLevel(logging.INFO)

    # Manually insert off-policy config
    n_control = dyna_config['drl']['config']['environment']['env_config']['act_dim']
    dyna_config['off_policy_pi'] = RandomPolicy(low=-1*np.ones(n_control), 
                                                high = np.ones(n_control), 
                                                seed=0)
    
    ip_head = os.environ.get('ip_head', None)
    ray.init(address=ip_head, 
             logging_level=logging.ERROR)
    print(ray.nodes())

    pbt_sched = None
    if dyna_config.get('use_pbt', False):
        pbt_config = dyna_config['pbt_config']
        hyper_mut = {}
        for key, val in pbt_config['hyperparam_mutations'].items():
            search_class = getattr(tune, val['search_class'])
            hyper_mut[key] = search_class(*val['search_space'])
        
        pbt_config['hyperparam_mutations'] = hyper_mut
            
        pbt_sched = PopulationBasedTraining(
                        **pbt_config,
                        custom_explore_fn=explore
                    )

    ray_config = dyna_config['ray_config']
    run_config=air.RunConfig(
        local_dir=LOCAL_DIR,
        **ray_config['run_config']
    )
    
    
    tune_config=tune.TuneConfig(
                    **dyna_config['ray_config']['tune_config'],
                    scheduler=pbt_sched
                    )
    
    drl_class, drl_default_config = rllib_algos.get(dyna_config['drl']['class'])()
    
    tune.Tuner(
        tune.with_resources(dyna_sindy, 
                            drl_class.default_resource_request(drl_default_config)
                            ),
        param_space=dyna_config, # this is what is passed to the experiment
        run_config=run_config,
        tune_config=tune_config
    ).fit()