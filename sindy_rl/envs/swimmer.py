from gym.envs.mujoco.swimmer_v4 import SwimmerEnv
import numpy as np


'''
Default Observation

    | Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | angle of the front tip               | -Inf | Inf | free_body_rot                    | hinge | angle (rad)              |
    | 1   | angle of the first rotor             | -Inf | Inf | motor1_rot                       | hinge | angle (rad)              |
    | 2   | angle of the second rotor            | -Inf | Inf | motor2_rot                       | hinge | angle (rad)              |
    | 3   | velocity of the tip along the x-axis | -Inf | Inf | slider1                          | slide | velocity (m/s)           |
    | 4   | velocity of the tip along the y-axis | -Inf | Inf | slider2                          | slide | velocity (m/s)           |
    | 5   | angular velocity of front tip        | -Inf | Inf | free_body_rot                    | hinge | angular velocity (rad/s) |
    | 6   | angular velocity of first rotor      | -Inf | Inf | motor1_rot                       | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of second rotor     | -Inf | Inf | motor2_rot                       | hinge | angular velocity (rad/s) |
'''

class SwimmerSurrogate(SwimmerEnv):
    '''
    Wrapper for Swimmer-v4 Environment [1]
    
        [1] https://github.com/openai/gym/blob/master/gym/envs/mujoco/swimmer_v4.py
    '''
    def __init__(self, env_config=None):
        env_kwargs = env_config.get('env_kwargs', {})

        # ENFORCE FULL OBSERVABILITY
        # env_kwargs.update({'exclude_current_positions_from_observation': False})
        
        
        # need these to standardize the environment.
        # quantifies when success is met.
        self.mod_angles = env_config.get('mod_angles', False)
        self.use_trig_obs = env_config.get('use_trig_obs', False)
        self.max_episode_steps = env_config.get('max_episode_steps', 1000)
        self.reward_threshold = env_config.get('reward_threshold', 360.0)

        super().__init__(**env_kwargs)
        
        # angle indices
        self.angle_idx =  np.array([0,1,2]) + 2*(1-self._exclude_current_positions_from_observation)

        # Whether or not to use the surrogate model
        self.dyn_model = env_config.get('dyn_model', None)
        
    def mod2pi(self, angles):
        return ((angles + np.pi) % (2*np.pi)) - np.pi
        
    def get_done(self):
        done = bool(
            # x < -self.x_threshold
            # or x > self.x_threshold
            # or theta < -self.theta_threshold_radians
            # or theta > self.theta_threshold_radians
            self.n_episode_steps >= self.max_episode_steps
            # or self.episode_reward >= self.reward_threshold
        )
        return done
        
    def get_reward(self, action, x_velocity):
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        return forward_reward - ctrl_cost
    
    def action_map(self, action):
        return action
    
    def real_step(self, action):
        xy_position_before = self.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = self._forward_reward_weight * x_velocity

        ctrl_cost = self.control_cost(action)

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        info = {
            "reward_fwd": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, info
    

    def trig_obs(self, state):
        new_state = np.zeros(len(state) + 3)
        angle_idx = np.array([0,1,2])
        
        if not self._exclude_current_positions_from_observation:
            angle_idx += 2
        
        # should work regardless of exlcuding pos
        new_state[:angle_idx[0]] = state[:angle_idx[0]]
        new_state[-5:] = state[-5:]
        
        new_state[angle_idx] = np.cos(state[angle_idx])
        new_state[angle_idx + 3] = np.sin(state[angle_idx])
        
        return new_state

    def step(self, action):
        if self.dyn_model:
            u = action
            # self.state = self.dyn_model.predict(self.prev_state, u)
            self.state = self.dyn_model.predict(self.state, u)
            
            # x_vel = (self.state[0] - self.prev_state[0])/self.dt
            
            # self.prev_state = self.state.copy()
            if self._exclude_current_positions_from_observation:
                x_vel = self.state[3]
            else: 
                x_vel = self.state[5]
            
            reward = self.get_reward(action, x_vel)
            info = {}
            
        else:
            self.state, reward, info = self.real_step(action)
            # if self.use_trig_obs:
            #     self.state = self.trig_obs(self.state)
        
        if self.mod_angles:
            self.state[self.angle_idx] = self.mod2pi(self.state[self.angle_idx])
        
        self.episode_reward += reward
        self.n_episode_steps += 1
        done = self.get_done()
        
        return np.array(self.state, dtype=np.float32), reward, done, info
    
    def reset(self, **reset_kwargs):
        '''
        Wraps reset
        '''
        # self.prev_state, _  = super().reset(**reset_kwargs)
        self.state, _ = super().reset(**reset_kwargs)
        self.n_episode_steps = 0
        self.episode_reward = 0
        
        if self.use_trig_obs:
            self.state = self.trig_obs(self.state)
        
        return self.state
        # return self.prev_state
    
if __name__ == '__main__': 
    env = SwimmerSurrogate(env_config = {'dyn_config': None, 
                                         'env_kwargs': 
                                                {'exclude_current_positions_from_observation': False
                                                 }, 
                                         'use_trig_obs': True
                                         })
    obs = env.reset(seed=0)
    print(len(env._get_obs()))
    print(len(obs))
    
    
    print(len(env.step(np.zeros(2))[0]))
    print(len(env.state))
    
    # new_obs = env.trig_obs(obs)
    # print(new_obs)