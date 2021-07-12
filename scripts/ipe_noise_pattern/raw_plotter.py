import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
import sys

root_dir =  f'/directory/to/quantum_bayesian_neural_networks'
quantum_simulations_dir = f'{root_dir}/scripts/quantum_simulation'

sys.path.insert(1, quantum_simulations_dir)

from quantum_computer import Quantum_Computer


def get_data(num_predict_data, seed):
    D_X = 2
    np.random.seed(seed)
    
    X_predict = np.random.rand(num_predict_data) * 2 - 1.0 # X_predict \in [-1, 1)

    X_predict = jnp.power(X_predict[:, np.newaxis], jnp.arange(D_X))

    W = jnp.array([0.5, 1.0])
    return X_predict, W


assert numpyro.__version__.startswith('0.6.0')
numpyro.set_platform('cpu')

ns = [2, 4, 6, 8, 10]
num_predict_data = 1000000

fontsize = 75

seed = 0
seed_data = 0

numpyro.set_host_device_count(1)

X_predict, W =  get_data(num_predict_data=num_predict_data, seed=seed_data)

rng_key, _ = random.split(random.PRNGKey(seed))

for n in ns :
    print('n: ' + str(n))
    quantum_computer = Quantum_Computer(n = n)
    
    
    vmap_args = (
        random.split(rng_key, num_predict_data),
        X_predict,
    )
    predictions = vmap(
        lambda rng_key_pred, x: quantum_computer.quantum_inner_product_estimation(rng_key_pred, x, W), in_axes=(0, 0), out_axes=0
    )(*vmap_args)

    # make plots
    fig, ax = plt.subplots(figsize=(40, 30), constrained_layout=True)

    # plot prediction data
    ax.scatter(X_predict[:, 1], predictions, c='red', alpha=0.01, s=1.0, marker=',')
    
    
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-0.6, 1.6])

    ax.tick_params(length=50.0, width=5.0, labelsize=fontsize)
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=8)
    
    ax.spines['bottom'].set_linewidth(5.0)
    ax.spines['left'].set_linewidth(5.0)
    ax.spines['top'].set_linewidth(5.0)
    ax.spines['right'].set_linewidth(5.0)
    
    ax.set_xlabel('X', fontsize=fontsize+20.0, labelpad=40.0)
    ax.set_ylabel('Y', rotation=0, va='center_baseline', fontsize=fontsize+20.0, labelpad=0.0)

    plt.savefig(f'{root_dir}/figs/ipe_noise_pattern/raw/n_{n}.png')
