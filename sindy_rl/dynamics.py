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
    Generic SINDy Dynamics model
    '''
    def __init__(self, dyn_config=None):
        self.config = dyn_config or {}
        self.dt = self.config.get('dt', 1)
        self.optimizer_name = self.config.get('optimizer', 'STLSQ')
        self.optimizer_kwargs = self.config.get('optimizer_kwargs', {})
        self.optimizer = getattr(ps, self.optimizer_name)(**self.optimizer_kwargs)

        self.model = ps.SINDy(discrete_time=True,optimizer=self.optimizer)
        
    
    def fit(self, observations, actions):
        self.model.fit(observations, u=actions, multiple_trajectories=True, t=self.dt)
        return self.model

    def predict(self, state, action):
        return self.model.simulate(x0=state, t=2, u=np.array([action]))[-1]

if __name__ == '__main__':
    from data_utils import collect_random_data
    from environment import CartSurrogate

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

    
    

    