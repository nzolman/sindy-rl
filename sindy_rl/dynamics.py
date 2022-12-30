import pysindy as ps
import numpy as np

class BaseDynModel:
    '''
    Base Dynamics Model. Every function needs
    '''

    def __init__(self):
        pass

    def predict(self, state, action):
        '''
        Should be over-written. Predicts next state given a (state, action) 
            combination.

        Parameters:
            - state: state of the system
            - action: the action that's taken
        '''
        return state


class CartPoleGymDynamics:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

    def predict(self, state, action):

        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        state = (x, x_dot, theta, theta_dot)

        return state


class SINDyDynamics:
    '''
    Generic discrete SINDy Dynamics model
    '''
    def __init__(self, dyn_config=None):
        self.config = dyn_config or {}
        self.dt = self.config.get('dt', 1)

        optimizer = self.config.get('optimizer', 'STLSQ')
        if type(optimizer) == str:
            self.optimizer_name = optimizer
            self.optimizer_kwargs = self.config.get('optimizer_kwargs', {})
            self.optimizer = getattr(ps, self.optimizer_name)(**self.optimizer_kwargs)
        else:
            self.optimizer = optimizer
        
        self.feature_library = self.config.get('feature_library', ps.PolynomialLibrary())
        
        self.model = ps.SINDy(discrete_time=True, 
                              optimizer=self.optimizer, 
                              feature_library=self.feature_library)

    def fit(self, observations, actions, **kwargs):
        self.model.fit(observations, u=actions, multiple_trajectories=True, t=self.dt, **kwargs)
        return self.model

    def predict(self, state, action):
        return self.model.simulate(x0=state, t=2, u=np.array([action]))[-1]


# TO-DO:
# 1. Configure bound_thresh
# 2. Configure stable_idx
# 3. Utilize stable_idx
class EnsembleSINDyDynamics(SINDyDynamics):
    '''
    Ensemble discrete SINDy Dynamics model
    '''
    def __init__(self, dyn_config=None):
        super().__init__(dyn_config=dyn_config)
        self.n_models = self.model.optimizer.n_models
        self.model_flags = np.zeros(self.n_models)
        self.bound_thresh = 100
        self.smart = dyn_config.get('use_smart', False)
        self.mean_coef =  np.mean(self.model.optimizer.coef_list, axis=0)
        
        self.stable_idx = np.arange(self.n_models)
        
        self.use_median = dyn_config.get('use_median', False)
        
    def predict(self, state, action):
        if self.smart:
            states = np.zeros((self.n_models, len(state)))
            for i in range(self.n_models):
                coef = self.model.optimizer.coef_list[i]
                self.model.optimizer.coef_ = coef
                states[i] = self.model.simulate(x0=state, t=2, u=np.array([action]))[-1]
                
            self.model_flags = np.any(np.abs(states) > self.bound_thresh, axis=1)
            
            trusted_states = states[~self.model_flags]
            assert len(trusted_states) > 0, "Ruh-roh"
            
            return np.median(trusted_states, axis=0)
        else:
            if self.use_median:
                coef = np.median(self.model.optimizer.coef_list, axis=0)
            else:
                coef = np.mean(self.model.optimizer.coef_list, axis=0)

            self.model.optimizer.coef_ = coef
            return self.model.simulate(x0=state, t=2, u=np.array([action]))[-1]
    
# ens_dyn_model = EnsembleSINDyDynamics(dyn_config=quad_affine_config)
# ens_dyn_model.optimizer = dyn_model.optimizer
# ens_dyn_model.model = dyn_model.model



if __name__ == '__main__':
    from data_utils import collect_random_data
    from sindy_rl.envs.cartpole import CartSurrogate

    true_dyn_model = CartPoleGymDynamics()
    env_config = {'dyn_model': true_dyn_model}
    env = CartSurrogate(env_config)
    env.reset()
    n_steps = int(1e4)
    trajs_action, trajs_obs = collect_random_data(env,n_steps)

    N_train_steps = 50
    tmp_steps = 0
    u_train = []
    x_train = []
    for i, (traj_a, traj_o) in enumerate(zip(trajs_action, trajs_obs)):
        n_steps = len(traj_a)
        if tmp_steps + n_steps < N_train_steps:
            print(i, N_train_steps, n_steps)
            u_train.append(np.array(traj_a))
            x_train.append(np.array(traj_o))
            tmp_steps += n_steps

        # if we go over, let's chop the last one
        else:
            diff_steps = N_train_steps - tmp_steps
            print(i, N_train_steps, diff_steps, n_steps)
            u_train.append(np.array(traj_a)[:diff_steps])
            x_train.append(np.array(traj_o)[:diff_steps])

            x_test = [np.array(traj) for traj in trajs_obs[i+1:]]
            u_test = [np.array(traj) for traj in trajs_action[i+1:]]
            x0_test = [x[0] for x in x_test]
            break

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

    dyn_model.model.print()
    score = dyn_model.model.score(x_test, u=u_test, multiple_trajectories=True)

    
    

    