def update_dyn_and_rew_models(dyn_model_weights, rew_model_weights):
    '''ray wrapper for updating surrogate models'''
    def update_env_models(env):
        env.update_models_(dyn_model_weights, rew_model_weights)
    
    def update_env_fn(worker):
        worker.foreach_env(update_env_models)

    return update_env_fn
    

def make_update_env_fn(env_conf):
    '''
    Updates the environment config
    Source [1] 
    
    [1] https://discuss.ray.io/t/update-env-config-in-workers/1831/3
    '''
    def update_env_conf(env):
        env.config.update(env_conf)
        env.game.configure(env.config)
        
    def update_env_fn(worker):
        worker.foreach_env(update_env_conf)

    return update_env_fn

def update_env_dyn_model(dyn_model):
    '''ray wrapper for setting the dynamics model'''
    def update_env(env):
        # TO-DO: Is setattr safer? 
        env.dyn_model = dyn_model
        
    def update_env_fn(worker):
        worker.foreach_env(update_env)
    
    return update_env_fn