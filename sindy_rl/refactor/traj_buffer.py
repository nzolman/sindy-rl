from collections import deque
import numpy as np
import pickle

class BaseTrajectoryBuffer:
    '''
    Intent is just lightweight bookkeeping for trajectories, mostly 
    wrappers for deques
    '''
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config
        self.max_traj = config.get('max_traj', None)
        
        self.x_traj_buffer = deque([], self.max_traj)
        self.u_traj_buffer = deque([], self.max_traj)
        self.rew_traj_buffer = deque([], self.max_traj)
        self.n_lens = deque([], self.max_traj)
        
    def __len__(self):
        return len(self.x_traj_buffer)
    
    def total_samples(self):
        '''Return total number of samples'''
        return np.sum(self.n_lens)

    def append(self, x, u, r  = None):
        '''Update all deques'''
        assert len(x) == len(u), f'x {(len(x))} and u {(len(u))} must have the same shape'
        self.x_traj_buffer.append(x)
        self.u_traj_buffer.append(u)
        self.rew_traj_buffer.append(r)
        self.n_lens.append(len(x))
        
    def popleft(self):
        '''Pop all deques from left. Return corresponding x, u, rew'''
        self.n_lens.popleft()
        x = self.x_traj_buffer.popleft()
        u = self.u_traj_buffer.popleft()
        r = self.rew_traj_buffer.popleft()
        return x, u, r
    
    def pop(self):
        '''Pop all deques. Return corresponding x, u'''
        self.n_lens.pop()
        x = self.x_traj_buffer.pop()
        u = self.u_traj_buffer.pop()
        r = self.rew_traj_buffer.pop()
        return x, u, r
    
    def to_list(self):
        '''Create lists from buffers'''
        return list(self.x_traj_buffer), list(self.u_traj_buffer), list(self.rew_traj_buffer)

    def to_dict(self):
        '''Create dictionary from buffers'''
        x, u, r = self.to_list()
        data = {'x': x, 'u': u, 'r': r}
        return data
    
    def add_data(self, x_data, u_data, r_data=None):
        '''
        Add data to the respective deques
        
        Inputs: 
            x_data: (iter)
                iterable of state data
            u_data: (iter)
                iterable of control data
        '''
        if r_data is None: 
            r_data = [None for x in x_data]
        for x, u,r in zip(x_data, u_data, r_data):
            self.append(x, u, r)
            
    def load_data(self, fname, clean = True):
        '''
        Inputs:
            fname: (str)
                filename
            clean: (bool)
                whether to reset the buffer before adding the data
        NOTE: 
            Still using the deque structure to add! If file contains
            more trajectories (or samples, etc.) than maximum, this
            will only contain the most recent!
        '''
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        if clean:
            self.x_traj_buffer = deque([], self.max_traj)
            self.u_traj_buffer = deque([], self.max_traj)
            self.rew_traj_buffer = deque([], self.max_traj)
            self.n_lens = deque([], self.max_traj)
        
        self.add_data(data['x'], data['u'], data['r'])
    
    def save_data(self, fname):
        '''Save data to filename'''
        data = self.to_dict()
        with open(fname, 'wb') as f:
            pickle.dump(data, f)

class MaxSamplesBuffer(BaseTrajectoryBuffer):
    '''
    Handle when there is a maximum number of total samples
    instead of just a maximal number of trajectories.
    '''
    def __init__(self, config=None):
        super().__init__(config)
        self.max_samples = config.get('max_samples', np.inf) or np.inf
    
    def append(self, x, u, r = None):
        n_samples = self.total_samples()
        tot_samples = n_samples + len(x)
        
        # determine which trajectories to get rid of
        if tot_samples > self.max_samples:
            diff = tot_samples - n_samples
            pop_idx = np.cumsum(self.n_lens) < diff
            for i in range(sum(pop_idx)+1):
                self.popleft()
        super().append(x,u, r)
