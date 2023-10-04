import warnings
warnings.filterwarnings('ignore')

import logging
import os
import numpy as np
from ray.rllib.algorithms.registry import ALGORITHMS as rllib_algos
from ray.tune.schedulers import PopulationBasedTraining
from ray import tune, air
from ray.air import session, Checkpoint
from ray.rllib.algorithms.algorithm import Algorithm

from sindy_rl.refactor.dyna import DynaSINDy


def dyna_sindy(config):
    dyna_config = config
    train_iterations = config['n_train_iter']
    dyn_fit_freq = config['dyn_fit_freq']
    ckpt_freq = config['ray_config']['checkpoint_freq']
    
    # data collected (or loaded) upon initialization
    dyna = DynaSINDy(dyna_config)
    
    if session.get_checkpoint():
        air_checkpoint = session.get_checkpoint()
        air_dict = air_checkpoint.to_dict()
        algo_check_path = air_dict['algo_path']

        dyna.load_checkpoint(algo_check_path)
        dyna.dynamics_model = air_dict['dyn_model']
        dyna.rew_model = air_dict['rew_model']
        dyna.on_policy_buffer = air_dict['on-policy']
        dyna.off_policy_buffer = air_dict['off-policy']
        dyna.update_surrogate()

    # setup the dynamics, reward, DRL algo push weights to surrogate
    # on remote workers
    dyna.fit_dynamics()
    dyna.fit_rew()
    dyna.update_surrogate()
    dyna.save_checkpoint(ckpt_num=-1, 
                         save_dir = tune.get_trial_dir())
    
    collect_dict = {'mean_rew': np.nan, 
                    'mean_len': 0}
    for n_iter in range(train_iterations):
        checkpoint = None
        train_results = dyna.train_algo()
        
        # Checkpoint
        if (n_iter % ckpt_freq) == ckpt_freq -1:
            algo_check_path = dyna.save_checkpoint(ckpt_num=n_iter, 
                                                   save_dir = tune.get_trial_dir()
                                                   )
            check_dict = {'algo_path': algo_check_path,
                          'dyn_model': dyna.dynamics_model,
                          'rew_model': dyna.rew_model,
                          'on-policy': dyna.on_policy_buffer,
                          'off-policy': dyna.off_policy_buffer}
            checkpoint = Checkpoint.from_dict(check_dict)
            
            
        if (n_iter % dyn_fit_freq) == dyn_fit_freq - 1:
            (trajs_obs, 
             trajs_acts, 
             trajs_rew) = dyna.collect_data(dyna.on_policy_buffer,
                                            dyna.real_env,
                                            dyna.on_policy_pi,
                                            **dyna_config['on_policy_buffer']['collect']
                                            )

            dyna.fit_dynamics()
            dyna.fit_rew()
            dyna.update_surrogate()
            
            collect_dict = {}
            collect_dict['mean_rew'] = np.mean([np.sum(rew) for rew in trajs_rew])
            collect_dict['mean_len'] = np.mean([len(obs) for obs in trajs_obs]) 
        train_results['traj_buffer'] = dyna.get_buffer_metrics()
        train_results['dyn_collect'] = collect_dict
        
        # TO-DO: determine if this is necessary.
        if checkpoint:
            session.report(train_results, 
                           checkpoint=checkpoint)
        else:
            session.report(train_results)
    
def explore(dyna_config):
    '''
    Used for PBT. 
    Ensures explored (continuous) parameters stay in a given range
    '''
    config = dyna_config['drl']['config']['training']
    
    config['lambda_'] = np.clip(config['lambda_'], 0, 1)
    config['gamma'] = np.clip(config['gamma'], 0, 1)
    
    dyna_config['drl']['config']['training'] = config
    return dyna_config

if __name__ == '__main__': 
    import yaml
    import logging
    import ray
    
    from pprint import pprint
    
    from sindy_rl.refactor.policy import RandomPolicy
    from sindy_rl import _parent_dir
    
    filename = '/home/firedrake/sindy-rl/sindy_rl/refactor/dyna_cylinder_test.yml'
    with open(filename, 'r') as f:
        dyna_config = yaml.load(f, Loader=yaml.SafeLoader)
    LOCAL_DIR =  os.path.join(_parent_dir, 'ray_results',dyna_config['exp_dir'])
    
    pprint(dyna_config)
    
    # Setup logger
    logging.basicConfig()
    logger = logging.getLogger('dyna-sindy')
    logger.setLevel(logging.INFO)

    # TO-DO: find a better place to automate this
    n_control = dyna_config['dynamics_model']['config']['feature_library']['kwargs']['n_control']
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