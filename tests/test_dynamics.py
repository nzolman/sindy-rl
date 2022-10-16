import numpy as np
import pysindy as ps
import warnings

from sindy_rl.data_utils import collect_random_data, split_by_steps
from sindy_rl.environment import CartSurrogate
from sindy_rl.dynamics import CartPoleGymDynamics, SINDyDynamics

def test_sindy_score():
    warnings.filterwarnings("ignore")
    SEED = 42
    np.random.seed(SEED)

    true_dyn_model = CartPoleGymDynamics()
    env_config = {'dyn_model': true_dyn_model}
    env = CartSurrogate(env_config)
    n_steps = int(1e4)
    trajs_action, trajs_obs = collect_random_data(env, n_steps, seed=SEED)
    N_train_steps = 50
    x_train, u_train, x_test, u_test = split_by_steps(N_train_steps, trajs_action, trajs_obs)

    dyn_config = {
                'dt': 0.02,
                'optimizer': 'STLSQ',
                'optimizer_kwargs': {
                    'threshold': 0.01,
                    'alpha': 0.5
                },
            }

    # Create and fit the SINDy Model
    dyn_model = SINDyDynamics(dyn_config)
    dyn_model.fit(observations = x_train, actions=u_train)
    score = dyn_model.model.score(x_test, u=u_test, multiple_trajectories=True)

    assert np.abs(score - 1) < 1e-5

if __name__ == '__main__':
    test_sindy_score()