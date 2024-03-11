import os
import pickle
import numpy as np
from copy import deepcopy

from sindy_rl import _parent_dir
from sindy_rl.env import rollout_env
from sindy_rl.policy import RLlibPolicyWrapper
from sindy_rl.sparse_policy.utils import get_models, fit_policies_v, hyper_mesh

import ray
try:
    ray.init(log_to_driver=False)
except Exception as e:
    print(e)


if __name__ == '__main__':
    trial_dir = os.path.join(_parent_dir, 'data/agents/swingup/dyna')
    
    # load models, data, and env
    cart_dyn, cart_rew, cart_data, cart_policy, cart_env = get_models(
                                                                        trial_dir,
                                                                        check_idx= 1150, # 1350, for overfit
                                                                        return_policy=True,
                                                                        )
    cart_unbounded_policy = RLlibPolicyWrapper(cart_policy.algo.get_policy(), mode='policy')


    # evaluate baseline
    cart_baseline_obs, cart_baseline_acts, cart_baseline_rews = rollout_env(
                                                                        cart_env.real_env, 
                                                                        policy=cart_policy, 
                                                                        n_steps = 15000, 
                                                                        verbose=True
                                                                        )
    baseline_perf = np.median([r.sum() for r in cart_baseline_rews])


    def fit_cart(env, dyn_model, policy, unbounded_policy, 
                hyper_mesh, n_dyn_rollout=50000, use_median=False, poly_deg = 3):
        '''
        Essentially a wrapper for fit_policies_v
        
        Arguments:
            env: sindy_rl.env.BaseEnsembleSurrogateEnv
                environment to run
            dyn_model: sindy_rl.dynamics.EnsembleSINDyDynamicsModel
                surrogate dynamics
            policy: sindy_rl.policy.RLlibPolicyWrapper
                policy to fit using rllib "algorithm" API, where actions are clipped
            unbounded_policy: sindy_rl.policy.RLlibPolicyWrapper
                policy using policy API, where actions are _NOT_ clipped
            hyper_mesh:
                np.meshgrid of hyper parameters to sweep over.
            n_dyn_rollout:
                number of surrogate experience points to generate
            use_median:
                whether to use the median ensemble coefficients (True) or the mean (False)
            poly_deg: maximal degree of polynomial for sparse policy.
        '''
        
        # randomly generate initial conditions near the stable equilibrium
        np.random.seed(0)
        N_ICs = 400
        th_eps = 0.1
        th_random = np.random.normal(loc = np.pi,
                                    scale=0.1,
                                    size = (N_ICs)
                                    )
        ic_X = np.random.normal(loc=0, scale=0.25,
                                size = (N_ICs, 5)
                                )
        ic_X[:,1] = np.cos(th_random)
        ic_X[:,2] = np.sin(th_random)
            
        # rollout trajectories using the SINDy dynamics
        env.dynamics_model = deepcopy(dyn_model)
        env.dynamics_model.set_median_coef_()
        env.use_real_env = False
        env.reset_from_buffer = True
        env.buffer_dict = {'x': [ic_X]} # force to draw from particular ICs
        env.reset()
        traj_obs, traj_acts, traj_rews = rollout_env(env, policy, 
                                                    n_steps = n_dyn_rollout, 
                                                    n_steps_reset=500, verbose = True)

        # randomly generate noise on top of the trajectories
        np.random.seed(1)
        n_resamples = 2
        X_data = np.concatenate(traj_obs)
        X_data = np.concatenate([X_data for n in range(n_resamples)])

        noise_scale = 1e-1
        noise =  np.random.normal(loc = 0,
                                    scale = noise_scale,
                                    size = X_data.shape)
        X_data = np.concatenate([
                                X_data + noise])

        # evaluate the NN control on these trajectories
        U_data = unbounded_policy.algo.compute_actions(X_data, explore=False)[0]
        U_data = np.clip(U_data, -5, 5)

        traj_X_normal = X_data
        traj_U_normal = U_data
        
        # evaluation data is just without noise (note, this might overfit---can instead split!)
        traj_X_eval = np.concatenate(traj_obs)
        traj_U_eval = unbounded_policy.algo.compute_actions(traj_X_eval, explore=False)[0]
        traj_U_eval = np.clip(traj_U_eval, -5, 5)
        
        
        normal_traj_data = {
                'X_data': traj_X_normal,
                'U_data': traj_U_normal,
                'X_eval': traj_X_eval,
                'U_eval': traj_U_eval,     
            }

        data_dicts = {
                        'normal_traj': normal_traj_data,
                    }

        # sweep through hyper params
        val = normal_traj_data
        CLIP_BOUNDS = np.array([[-1.0], [1.0]])
        res_dict = fit_policies_v(**val, hyper_mesh=hyper_mesh, n_models = 20, 
                                use_median = use_median, clip_params = CLIP_BOUNDS, bounds = CLIP_BOUNDS, 
                                poly_deg = poly_deg)
        sparse_policy = res_dict['best_policy']
        sparse_policy.set_median_coef_()
        
        return res_dict, normal_traj_data


    # fit the policies (sweep through hyper parameters)
    # and select best one
    cart_res_dict, cart_normal_traj_data = fit_cart(cart_env, cart_dyn, 
                                                cart_policy, cart_unbounded_policy, 
                                                hyper_mesh,
                                                n_dyn_rollout=5000,
                                                use_median=False,
                                                poly_deg=3)

    cart_sparse_policy = cart_res_dict['best_policy']

    # example of saving the policy
    with open('cart_sparse_policy.pkl', 'wb') as f:
        pickle.dump(cart_sparse_policy, f)

    cart_sparse_policy.set_mean_coef_()
    cart_eval_obs, cart_eval_acts, cart_eval_rews = rollout_env(cart_env.real_env, 
                                                                policy=cart_sparse_policy, 
                                                                n_steps = 15000, verbose=True)
    perf = np.median([r.sum() for r in cart_eval_rews])

    # print comparison of rewards
    print(f'baseline: {baseline_perf}, sparse: {perf}')

