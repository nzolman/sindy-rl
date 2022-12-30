import numpy as np

from gym.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
'''
    ### Action Space
    The agent take a 1-element vector for actions.
    The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
    the numerical force applied to the cart (with magnitude representing the amount of
    force and sign representing the direction)
    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |
    ### Observation Space
    The state space consists of positional values of different body parts of
    the pendulum system, followed by the velocities of those individual parts (their derivatives)
    with all the positions ordered before all the velocities.
    The observation is a `ndarray` with shape `(4,)` where the elements correspond to the following:
    | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Unit                      |
    | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
    | 0   | position of the cart along the linear surface | -Inf | Inf | slider                           | slide | position (m)              |
    | 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge                            | hinge | angle (rad)               |
    | 2   | linear velocity of the cart                   | -Inf | Inf | slider                           | slide | velocity (m/s)            |
    | 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge                            | hinge | anglular velocity (rad/s) |
    ### Rewards
    The goal is to make the inverted pendulum stand upright (within a certain angle limit)
    as long as possible - as such a reward of +1 is awarded for each timestep that
    the pole is upright.
'''

class InvPendSurrogate(InvertedPendulumEnv):
    '''
    Wrapper for Inverted-Pendulum-v4 Environment [1]
    
        [1] https://github.com/openai/gym/blob/6a04d49722724677610e36c1f92908e72f51da0c/gym/envs/mujoco/inverted_pendulum_v4.py
    '''
    def __init__(self, env_config=None):
        env_kwargs = env_config.get('env_kwargs', {})
        self.reward_threshold = env_config.get('reward_threshold', 950.0)
        self.max_episode_steps = env_config.get('max_episode_steps', 1000)
        self.angle_thresh = env_config.get('angle_thresh', 0.2)
        self.use_surrogate_rew = env_config.get('use_surrogate_rew', False)
        
        super().__init__(**env_kwargs)
        # Whether or not to use the surrogate model
        self.dyn_model = env_config.get('dyn_model', None)
    
    def get_reward(self):
        if self.use_surrogate_rew:
            sq_error = np.linalg.norm(self.state)**2 + 999*self.state[1]**2
            return  1 - 0.01*sq_error - 0.01*np.linalg.norm(self.action)**2
        else:
            return 1
    
    def get_done(self, ob):
        exceeded = self.n_episode_steps >= self.max_episode_steps
        return exceeded or bool(not np.isfinite(ob).all() or (np.abs(ob[1]) > self.angle_thresh))
    
    def real_step(self, action):
        reward = 1.0
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        terminated = self.get_done(ob)
        if self.render_mode == "human":
            self.render()
        
        return ob, reward, terminated, {}
    
    def step(self, action):
        self.action = action
        if self.dyn_model:
            u = action
            self.state = self.dyn_model.predict(self.state, u)
            reward = self.get_reward()
            info = {}
        else:
            self.state, reward, done, info = self.real_step(action)
        
        self.episode_reward += reward
        self.n_episode_steps += 1
        done = self.get_done(self.state)
        
        return np.array(self.state, dtype=np.float32), reward, done, info
    
    def reset(self, **reset_kwargs):
        '''
        Wraps reset
        '''
        self.state, _ = super().reset(**reset_kwargs)
        self.n_episode_steps = 0
        self.episode_reward = 0
        
        return self.state
    
    def action_map(self, action):
        '''
        Allow policy to be definied between [-1,1]
        '''
        return action
        # return 3 * action
