from matplotlib import pyplot as plt
import numpy as np

def plot_phase_cartpole(dyn_model, fig_label):
    fig, axes = plt.subplots(2,2, figsize=(7,7))
    edge = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(edge, edge)
    Z = np.zeros((100,100))
    Z1 = np.ones((100,100))

    idx = 0
    init_points = np.array([X.flatten(), Y.flatten(), Z.flatten(), Z.flatten()]).T
    dx = dyn_model.model.predict(init_points, u = np.zeros(len(init_points)))
    U = dx[:,0].reshape(100,100)
    V = dx[:,1].reshape(100,100)
    axes[0][idx].streamplot(X, Y, U, V, density=1.)
    axes[0][idx].set_ylabel(r'$\dot{x}(t)$', fontsize=12)
    axes[0][idx].set_xlabel(r'$x(t)$', fontsize=12)

    idx = 1
    init_points = np.array([Z.flatten(), Z.flatten(), X.flatten(), Y.flatten()]).T
    dx = dyn_model.model.predict(init_points, u = np.zeros(len(init_points)))
    U = dx[:,2].reshape(100,100)
    V = dx[:,3].reshape(100,100)
    axes[0][idx].streamplot(X, Y, U, V, density=1)
    axes[0][idx].set_ylabel(r'$\dot{\theta}(t)$', fontsize=12)
    axes[0][idx].set_xlabel(r'$\theta(t)$', fontsize=12)

    idx = 0
    init_points = np.array([X.flatten(), Y.flatten(), Z1.flatten(), Z1.flatten()]).T
    dx = dyn_model.model.predict(init_points, u = np.zeros(len(init_points)))
    U = dx[:,0].reshape(100,100)
    V = dx[:,1].reshape(100,100)
    axes[1][idx].streamplot(X, Y, U, V,density=1.)
    axes[1][idx].set_ylabel(r'$\dot{x}(t)$', fontsize=12)
    axes[1][idx].set_xlabel(r'$x(t)$', fontsize=12)

    idx = 1
    init_points = np.array([Z1.flatten(), Z1.flatten(), X.flatten(), Y.flatten()]).T
    dx = dyn_model.model.predict(init_points, u = np.zeros(len(init_points)))
    U = dx[:,2].reshape(100,100)
    V = dx[:,3].reshape(100,100)
    axes[1][idx].streamplot(X, Y, U, V, density=1)
    axes[1][idx].set_ylabel(r'$\dot{\theta}(t)$', fontsize=12)
    axes[1][idx].set_xlabel(r'$\theta(t)$', fontsize=12)

    fig.suptitle(fig_label, fontsize=15)


# ------------------------------------------------------------


def plot_train_trajs_cartpole(x_train, x_test, fig_label = 'Random Training/Test Data'):
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes = axes.flatten()

    idx = 0
    for i, x in enumerate(x_test):
        label = None
        if i==0: 
            label = 'test'
        axes[idx].plot(x[:,0], x[:,1], c = 'r', label = label, alpha = 0.1)

    for i, x in enumerate(x_train):
        label = None
        if i==0: 
            label = 'train'
        axes[idx].plot(x[:,0], x[:,1], c = 'b', label = label)

    axes[idx].legend()
    axes[idx].set_ylabel(r'$\dot{x}(t)$', fontsize=12)
    axes[idx].set_xlabel(r'$x(t)$', fontsize=12)

    idx = 1
    for i, x in enumerate(x_test):
        label = None
        if i==0: 
            label = 'test'
        axes[idx].plot(x[:,2], x[:,3], c = 'r', label = label, alpha = 0.1)

    for i, x in enumerate(x_train):
        label = None
        if i==0: 
            label = 'train'
        axes[idx].plot(x[:,2], x[:,3], c = 'b', label = label)

    axes[idx].legend()
    axes[idx].set_ylabel(r'$\dot{\theta}(t)$', fontsize=12)
    axes[idx].set_xlabel(r'$\theta(t)$', fontsize=12)

    fig.suptitle(fig_label, fontsize=15)
