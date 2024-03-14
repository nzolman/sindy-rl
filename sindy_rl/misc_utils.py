import io
import pickle


class RenameUnpickler(pickle.Unpickler):
    '''
    Taken from here: 
        https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
    for dealing with pickle when you have changed directory names
    '''
    def find_class(self, module, name):
        renamed_module = module
        
        if 'sindy_rl.refactor' in module: 
            mod_split = module.split('.')
            mod = mod_split[-1]
            if len(mod_split) == 3: 
                renamed_module = f'sindy_rl.{mod}'

        return super(RenameUnpickler, self).find_class(renamed_module, name)

def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()

def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)


if __name__ == '__main__':
    import os
    import glob
    from sindy_rl import _parent_dir
    
    root_dir = os.path.join(_parent_dir, 'data', 'agents')
    fnames = ['off-pi_data.pkl', 'on-pi_data.pkl', 
              'rew_model.pkl', 'dyn_model.pkl',
              'traj_eval-fine.pkl', 'traj_eval-med.pkl',
              'algorithm_state.pkl', 'policy_state.pkl',
              'sparse_policy.pkl', 
              'sparse_policy_OVERFIT.pkl']
    
    # safe resave of all the pickle files.
    for fname in fnames:
        fpaths = glob.glob(os.path.join(root_dir, '**',  fname), recursive=True)
        
        for fpath in fpaths:
            print(fpath)
            # safe load
            with open(fpath, 'rb') as f:
                obj = renamed_load(f)
            # resave
            with open(fpath, 'wb') as f:
                pickle.dump(obj, f)
        
    