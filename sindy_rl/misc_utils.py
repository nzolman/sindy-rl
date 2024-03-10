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