import numpy as np
from gym import logger
from gym.envs.classic_control.cartpole import CartPoleEnv

from sindy_rl.dynamics import BaseDynModel, SINDyDynamics, CartPoleGymDynamics


#--------------------------------
# TO-DO :
# - Make sure to catch any errors from pysindy! Would suck to blow-up in finite time.
#--------------------------------


class CartSurrogate(CartPoleEnv):
    '''
    Placeholder Environment to wrap the CartPole problem and enable the 
        abiltity to exchange different dynamics engines.
    '''
    def __init__(self, env_config = None):
        env_kwargs = env_config.get('env_kwargs', {})

        # need these to standardize the environment.
        # quantifies when success is met.
        self.max_episode_steps = env_config.get('max_episode_steps', 500)
        self.reward_threshold = env_config.get('reward_threshold', 475)

        super().__init__(**env_kwargs)
        self.dyn_model = env_config['dyn_model']

    def get_reward(self, done):
        if not done:
            reward = 1.0
        # else:
        #     reward = 0.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0
        
        return reward

    def get_done(self):
        '''
        Determine whether the episode has terminated.
        '''
        x, x_dot, theta, theta_dot = self.state

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.n_episode_steps >= self.max_episode_steps
            # or self.episode_reward >= self.reward_threshold
        )
        return done

    def action_map(self, action):
        '''
        Need to map from discrete variables to something more "physical".
            Namely, easier to think about as (-1, 1) instead of (0, 1).
        '''

        return (action - 0.5)*2

    def step(self, action):
        u = self.action_map(action)
        self.state = self.dyn_model.predict(np.array(self.state), u)
        
        self.n_episode_steps += 1
        done = self.get_done()

        reward = self.get_reward(done)
        self.episode_reward += reward

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self, **reset_kwargs):
        '''
        Wraps reset
        '''
        obs, _ = super().reset(**reset_kwargs)
        self.n_episode_steps = 0
        self.episode_reward = 0
        return obs

if __name__ == '__main__': 

    dyn_model = CartPoleGymDynamics()
    env_config = {'dyn_model': dyn_model}
    env = CartSurrogate(env_config)
    env2 = CartPoleEnv()

    observation= env.reset(seed=42)
    obs2 = env2.reset(seed=42)

    n_steps = 100
    for i in range(n_steps):
        action = env.action_space.sample()
        observation, reward, terminated, info = env.step(action)
        obs2, rew2, term2,_, info2 = env2.step(action)
        print(f'{i} obs: {observation}')
        print(f'{i} obs: {obs2} (gym)')
        print(f'{i} rew: {reward}')
        print(f'{i} rew: {rew2} (gym)')
        if terminated:
            observation = env.reset()
        if term2:
            obs2 = env2.reset()
    env.close()