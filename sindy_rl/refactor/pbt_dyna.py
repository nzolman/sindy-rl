import warnings
warnings.filterwarnings('ignore')

import logging
import os
import numpy as np
from ray.rllib.algorithms.registry import ALGORITHMS as rllib_algos
from ray.tune.schedulers import PopulationBasedTraining
from ray import tune, air
from ray.air import session, Checkpoint
# from ray import tune, train
# from ray.train import Checkpoint
from ray.rllib.algorithms.algorithm import Algorithm

from sindy_rl.refactor.dyna import DynaSINDy
from sindy_rl.refactor.env import rollout_env

import tempfile


def dyna_sindy(config):
    dyna_config = config
    train_iterations = config['n_train_iter']
    dyn_fit_freq = config['dyn_fit_freq']
    ckpt_freq = config['ray_config']['checkpoint_freq']
    
    # data collected (or loaded) upon initialization
    # TO-DO: this should depend on whether there's a checkpoint, probably. 
    dyna = DynaSINDy(dyna_config)
    
    start_iter = 0
    
    checkpoint = session.get_checkpoint()
    if checkpoint:
        
        check_dict = checkpoint.to_dict()
        checkpoint_dir = check_dict['checkpoint_dir']
        algo_dir = check_dict['algo_dir']
        dyna.load_checkpoint(check_dict)
        
        # grab the iteration to make sure we _probably_ are checkpointing correctly again...? 
        start_iter = check_dict['epoch'] + 1
        print('\n!!! LOADING !!!', start_iter, session.get_trial_id(), checkpoint_dir, algo_dir)

        
    # setup the dynamics, reward, DRL algo push weights to surrogate
    # on remote workers
    dyna.fit_dynamics()
    dyna.fit_rew()
    dyna.update_surrogate()
    
    
    # # !!!EXPERIMENTAL!!!
    # eval_freq = dyna_config['drl']['config']['evaluation']['evaluation_interval']
    # num_eval = dyna_config['drl']['config']['evaluation']['evaluation_duration']

    
    # custom_eval_dict = {'mean_rew': np.nan,
    #                     'mean_len': 0}

    # TO-DO: figure out the proper way to deal with this
    collect_dict = {'mean_rew': np.nan, 
                    'mean_len': 0}
    
    for n_iter in range(start_iter, train_iterations):
        print('\n!!!n_iter!!', n_iter, session.get_trial_id())
        checkpoint = None
        train_results = dyna.train_algo()
        
        
        # # !!!EXPERIMENTAL!!!
        # if n_iter % eval_freq == eval_freq - 1:
        #     dyna.real_env.reset_on_bounds = False
        #     eval_obs, eval_acts, eval_rews = rollout_env(dyna.real_env, dyna.on_policy_pi, n_steps = num_eval * 1000)
            
        #     custom_eval_dict['mean_rew'] = np.mean([rew.sum() for rew in eval_rews])
        #     custom_eval_dict['mean_len'] = np.mean([len(rew) for rew in eval_rews])
        #     dyna.real_env.reset_on_bounds = True


        # periodically collect data
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
        if ((n_iter % ckpt_freq) == ckpt_freq - 1) and (n_iter!=0):
            
            check_dict = dyna.save_checkpoint(ckpt_num=n_iter, 
                                              save_dir = session.get_trial_dir(),
                                                )
            checkpoint = Checkpoint.from_dict(check_dict)
        
        
        
        # train_results['custom_eval'] = custom_eval_dict
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
    
    # filename = '/home/firedrake/sindy-rl/sindy_rl/refactor/dyna_cylinder_test.yml'
    # filename = '/home/nzolman/projects/sindy-rl/sindy_rl/refactor/dyna_config_cart_nn_test.yml'
    filename = '/home/nzolman/projects/sindy-rl/sindy_rl/refactor/dyna_config_cart_test.yml'
    # filename = '/home/nzolman/projects/sindy-rl/sindy_rl/refactor/dyna_config_swimmer_test.yml'
    
    with open(filename, 'r') as f:
        dyna_config = yaml.load(f, Loader=yaml.SafeLoader)
    LOCAL_DIR =  os.path.join(_parent_dir, 'ray_results',dyna_config['exp_dir'])
    
    pprint(dyna_config)
    
    # Setup logger
    logging.basicConfig()
    logger = logging.getLogger('dyna-sindy')
    logger.setLevel(logging.INFO)

    # TO-DO: find a better place to automate this
    # n_control = dyna_config['dynamics_model']['config']['feature_library']['kwargs']['n_control']
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
        # storage_path=LOCAL_DIR,
        # checkpoint_config=air.CheckpointConfig(
        #     **ray_config.get('checkpoint_config', {})
        # ),
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