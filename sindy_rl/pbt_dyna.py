import warnings
warnings.filterwarnings('ignore')
import logging
import os
import numpy as np

from ray.rllib.algorithms.registry import ALGORITHMS as rllib_algos
from ray.tune.schedulers import PopulationBasedTraining
from ray import tune, air
from ray.air import session, Checkpoint

from sindy_rl.dyna import DynaSINDy


def dyna_sindy(config):
    '''
    ray.Tune functional API for defining an expirement
    '''
    dyna_config = config
    train_iterations = config['n_train_iter']
    dyn_fit_freq = config['dyn_fit_freq']
    ckpt_freq = config['ray_config']['checkpoint_freq']
    
    # data collected (or loaded) upon initialization
    dyna = DynaSINDy(dyna_config)
    
    start_iter = 0
    
    # for PBT, session is populated with a checkpoint after evaluating the population
    # and pruning the bottom performers
    checkpoint = session.get_checkpoint()
    if checkpoint:
        check_dict = checkpoint.to_dict()
        dyna.load_checkpoint(check_dict)
        
        # grab the iteration to make sure we are checkpointing correctly
        start_iter = check_dict['epoch'] + 1

        
    # setup the dynamics, reward, DRL algo push weights to surrogate
    # on remote workers
    dyna.fit_dynamics()
    dyna.fit_rew()
    dyna.update_surrogate()

    collect_dict = {'mean_rew': np.nan, 
                    'mean_len': 0}
    
    # Main training loop
    for n_iter in range(start_iter, train_iterations):
        checkpoint = None
        train_results = dyna.train_algo()

        # periodically evaluate by collecting on-policy data
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
            
        # Checkpoint (ideally after the latest collection)
        if ((n_iter % ckpt_freq) == ckpt_freq - 1):
            
            check_dict = dyna.save_checkpoint(ckpt_num=n_iter, 
                                              save_dir = session.get_trial_dir(),
                                                )
            checkpoint = Checkpoint.from_dict(check_dict)
        
        # compile metrics for tune to report
        train_results['traj_buffer'] = dyna.get_buffer_metrics()
        train_results['dyn_collect'] = collect_dict
        
        # may not be entirely necessay, copied from some official Tune
        #   pbt example
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
    
    from sindy_rl.policy import RandomPolicy
    from sindy_rl import _parent_dir
    
    # TO-DO: replace with argparse
    filename = os.path.join(_parent_dir, 
                            'sindy_rl',
                            'config_templates', 
                            'dyna_cylinder.yml' # replace with appropriate config yaml
                            )
    
    # load config
    with open(filename, 'r') as f:
        dyna_config = yaml.load(f, Loader=yaml.SafeLoader)
    LOCAL_DIR =  os.path.join(_parent_dir, 'ray_results',dyna_config['exp_dir'])
    pprint(dyna_config)
    
    # Setup logger
    logging.basicConfig()
    logger = logging.getLogger('dyna-sindy')
    logger.setLevel(logging.INFO)

    # Initialize default off-policy for initial collection
    n_control = dyna_config['drl']['config']['environment']['env_config']['act_dim']
    dyna_config['off_policy_pi'] = RandomPolicy(low=-1*np.ones(n_control), 
                                                high = np.ones(n_control), 
                                                seed=0)
    
    # setup ray
    ip_head = os.environ.get('ip_head', None)
    ray.init(address=ip_head, 
             logging_level=logging.ERROR)

    # PBT-specific config
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

    # ray + tune configs
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