import numpy as np

def collect_random_data(env, n_steps, seed=42):
    actions = []
    observation = env.reset(seed=seed)
    env.action_space.seed(seed)
    observations = [observation]
    trajs_action = []
    trajs_obs = []

    for i in range(n_steps):
        action = env.action_space.sample()
        actions.append(env.action_map(action))
        observation, reward, done, info = env.step(action)
        observations.append(observation)
        if done:
            observation = env.reset()
            observations.pop()
            trajs_action.append(np.array(actions))
            actions = []
            trajs_obs.append(np.array(observations))
            observations = [observation]

    return trajs_action, trajs_obs


def split_by_steps(N_train_steps, trajs_action, trajs_obs):
    u_train = []
    x_train = []
    tmp_steps = 0

    for i, (traj_a, traj_o) in enumerate(zip(trajs_action, trajs_obs)):
        n_steps = len(traj_a)
        if tmp_steps + n_steps < N_train_steps:
            u_train.append(np.array(traj_a))
            x_train.append(np.array(traj_o))
            tmp_steps += n_steps

        # if we go over, let's chop the last one
        else:
            diff_steps = N_train_steps - tmp_steps
            u_train.append(np.array(traj_a)[:diff_steps])
            x_train.append(np.array(traj_o)[:diff_steps])

            x_test = [np.array(traj) for traj in trajs_obs[i+1:]]
            u_test = [np.array(traj) for traj in trajs_action[i+1:]]
            break

    return x_train, u_train, x_test, u_test
