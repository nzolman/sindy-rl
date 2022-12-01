import ray
import ray.rllib.algorithms.ppo as ppo
from ray.tune.logger import pretty_print
from ray import tune, air
from pprint import pprint
from tqdm import tqdm

from sindy_rl.dynamics import SINDyDynamics, CartPoleGymDynamics
from sindy_rl.envs.cartpole import CartSurrogate
from sindy_rl.data_utils import collect_random_data, split_by_steps

def evaluate_model(config, n_resets, checkpoint_path):
    env_class = config['env']
    
    agent = ppo.PPO(config=config, env=env_class)
    agent.restore(checkpoint_path)

    env = env_class(config['env_config'])
    lens = []
    rews = []
    for i in tqdm(range(n_resets)):
        obs = env.reset(seed=i)
        tot_reward = 0
        for ii in range(env.max_episode_steps):
            act = agent.compute_single_action(obs, explore=False)
            obs, rew, done, info = env.step(act)
            tot_reward += rew
            if done:
                break
        lens.append(ii)
        rews.append(tot_reward)

    result = {
                'rews': rews,
                'lens': lens
            }
    return result


def load_dynamics_model(path):
    with open(path, 'rb') as f: 
        d = json.load(f)
    
    config = d.get('dyn_experiment_config', None)
    seed = config['collect_seed']
    dyn_config = config['dyn_model_config']

    real_env_config = {'dyn_model': CartPoleGymDynamics()}
    real_env = CartSurrogate(real_env_config)

    N_steps_collect = config['N_steps_collect']
    N_steps_train = config['N_steps_train']
    trajs_action, trajs_obs = collect_random_data(real_env, N_steps_collect, seed=seed)
    x_train, u_train, x_test, u_test = split_by_steps(N_steps_train, trajs_action, trajs_obs)

    # Train Dynamics Model
    # pprint(dyn_config)
    dyn_model = SINDyDynamics(dyn_config = dyn_config)
    
    dyn_model.fit(observations = x_train, actions=u_train)

    return dyn_model, x_train, u_train, x_test, u_test
