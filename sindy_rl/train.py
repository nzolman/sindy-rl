import ray
import ray.rllib.algorithms.ppo as ppo
from ray.tune.logger import pretty_print
from ray import tune, air
from pprint import pprint

from sindy_rl.dynamics import CartPoleGymDynamics, SINDyDynamics
from sindy_rl.data_utils import collect_random_data, split_by_steps
from sindy_rl.environment import CartSurrogate

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config['framework'] = 'torch'

real_env_config = {'dyn_model': CartPoleGymDynamics()}
real_env = CartSurrogate(real_env_config)

# Collect Data
SEED = 42
n_steps = int(1e4)
trajs_action, trajs_obs = collect_random_data(real_env, n_steps, seed=SEED)
N_train_steps = 25
x_train, u_train, x_test, u_test = split_by_steps(N_train_steps, trajs_action, trajs_obs)

# Train Dynamics Model
dyn_config = {
            'dt': 0.02,
            'optimizer': 'STLSQ',
            'optimizer_kwargs': {
               'threshold': 0.01,
               'alpha': 0.5
            },
      }
dyn_model = SINDyDynamics(dyn_config)
dyn_model.fit(observations = x_train, actions=u_train)
#  score = dyn_model.model.score(x_test, u=u_test, multiple_trajectories=True)

config['env'] = CartSurrogate
config['env_config'] = {'dyn_model': dyn_model}
# config['env_config'] = real_env_config


# pprint(config)



# algo = ppo.PPO(config=config)


# check_freq = 100
# for i in range(10):
#    # Perform one iteration of training the policy with PPO
#    result = algo.train()
#    print(pretty_print(result))

#    if i % check_freq ==0:
#       checkpoint = algo.save()
reward_threshold = 475
results = tune.Tuner(
                     ppo.PPO,
                     param_space=config,
                     run_config=air.RunConfig(
                        local_dir='~/ray_results/sindy_25/',
                        stop={'episode_reward_mean': reward_threshold},
                        checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True, checkpoint_frequency=10),
                     )).fit()


# algo.compute_single_action(obs)