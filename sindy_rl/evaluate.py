import ray
import ray.rllib.algorithms.ppo as ppo
from ray.tune.logger import pretty_print
from ray import tune, air
from pprint import pprint
from tqdm import tqdm

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

