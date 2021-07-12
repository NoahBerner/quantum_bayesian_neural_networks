import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp
import jax.random as random

import numpyro
import argparse
import sys

root_dir =  f'/directory/to/quantum_bayesian_neural_networks'
quantum_simulations_dir = f'{root_dir}/scripts/quantum_simulation'

sys.path.insert(1, quantum_simulations_dir)

from quantum_bayesian_neural_network import QBNN

from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import seaborn as sns

matplotlib.use('Agg')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_data(num_train_data, D_X, num_predict_data, seed):
    num_total = num_train_data + num_predict_data
    D_Y = 1  # create 1d outputs
    
    X, Y = make_moons(noise=0.2, random_state=seed, n_samples=num_total)

    X = scale(X)
    X = jnp.array(X)
    Y = jnp.array(Y)
    X_train, X_predict, Y_train, Y_predict = train_test_split(X, Y, test_size=num_predict_data, random_state=seed)

    Y_train = Y_train.reshape((num_train_data, D_Y))
    
    assert X_train.shape == (num_train_data, D_X)
    assert Y_train.shape == (num_train_data, D_Y)
    assert X_predict.shape == (num_predict_data, D_X)
    
    return X_train, Y_train, X_predict, Y_predict


assert numpyro.__version__.startswith('0.6.0')
numpyro.set_platform('cpu')  # change this to 'gpu' if available.

parser = argparse.ArgumentParser(description='Parses input variables for binary classification task on quantum BNN.')

parser.add_argument('-ci', '--classical_inference', type=str2bool, nargs='?', required=True, help='Use classical inference instead of quantum inference.')
parser.add_argument('-cp', '--classical_predict', type=str2bool, nargs='?', required=True, help='Use classical prediction instead of quantum prediction.')
parser.add_argument('-lri', '--low_rank_initialization', type=str2bool, nargs='?', required=True, help='Use low-rank initialization instead of full-rank initialization.')
parser.add_argument('-n', type=int, nargs='?', required=True, help='Number of qubits used in phase estimation simulation.')
parser.add_argument('--num_hidden', type=int, nargs='?', required=True, help='Number of neurons in the two hidden layers.')
parser.add_argument('--seed', type=int, nargs='?', required=True, help='Seed used for model randomness.')
args = parser.parse_args()

classical_inference = args.classical_inference
classical_predict = args.classical_predict
low_rank_initialization = args.low_rank_initialization
n = args.n
num_hidden = args.num_hidden

print(f'classical_inference: {classical_inference}')
print(f'classical_predict: {classical_predict}')
print(f'low_rank_initialization: {low_rank_initialization}')
print(f'n: {n}')
print(f'num_hidden: {num_hidden}')
print(f'seed: {args.seed}')

isClassifier = True
num_samples = 2000
num_warmup = 1000
num_chains = 1
num_hidden = 5
D_X = 2
num_train_data = 100
num_predict_data = 500

grid_resolution = 100
fontsize = 25
linewidth = 2.0

seed = args.seed
seed_data = args.seed
rng_key, rng_key_inference = random.split(random.PRNGKey(seed))
rng_key_grid_predict, rng_key_point_predict = random.split(rng_key)

numpyro.set_host_device_count(num_chains)

prefix_str = ''
if classical_inference :
    prefix_str += 'ci'
else :
    prefix_str += 'qi'
if classical_predict :
    prefix_str += 'cp'
else :
    prefix_str += 'qp'
if not classical_inference or not classical_predict :
    prefix_str += '_n_' + str(n)
    
rank_str = 'fr'
if low_rank_initialization :
    rank_str = 'lr'
    
X_train, Y_train, X_predict, Y_predict =  get_data(num_train_data=num_train_data, D_X=D_X, num_predict_data=num_predict_data, seed=seed_data)

qbnn = QBNN(n=n, classical_inference=classical_inference, classical_predict=classical_predict, low_rank_initialization=low_rank_initialization, isClassifier=isClassifier, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains, num_hidden=num_hidden)
    
samples = qbnn.run_inference(rng_key_inference, X_train, Y_train)

point_predictions = qbnn.predict(rng_key_point_predict, samples, X_predict)
mean_point_predictions = point_predictions.mean(0) > 0.5

imag_grid_resolution = grid_resolution * 1j
grid = jnp.array(np.mgrid[-3:3:imag_grid_resolution, -3:3:imag_grid_resolution])
grid_2d = grid.reshape(2, -1).T
grid_predictions = qbnn.predict(rng_key_grid_predict, samples, grid_2d)


#Plotting mean
cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

contour = ax.contourf(grid[0], grid[1], grid_predictions.mean(axis=0).reshape(grid_resolution, grid_resolution), cmap=cmap)

X_predict_0_correct = X_predict[[pred == 0 and corr == 0 for pred, corr in zip(mean_point_predictions, Y_predict)], :]
X_predict_0_incorrect = X_predict[[pred == 0 and corr == 1 for pred, corr in zip(mean_point_predictions, Y_predict)], :]
X_predict_1_correct = X_predict[[pred == 1 and corr == 1 for pred, corr in zip(mean_point_predictions, Y_predict)], :]
X_predict_1_incorrect = X_predict[[pred == 1 and corr == 0 for pred, corr in zip(mean_point_predictions, Y_predict)], :]

ax.scatter(X_predict_0_correct[:, 0], X_predict_0_correct[:, 1], color='tab:blue', label='Class 0 Prediction Correct')
ax.scatter(X_predict_0_incorrect[:, 0], X_predict_0_incorrect[:, 1], color='tab:cyan', label='Class 0 Prediction Incorrect')
ax.scatter(X_predict_1_correct[:, 0], X_predict_1_correct[:, 1], color='tab:red', label='Class 1 Prediction Correct')
ax.scatter(X_predict_1_incorrect[:, 0], X_predict_1_incorrect[:, 1], color='tab:pink', label='Class 1 Prediction Incorrect')



ax.set_xlim([-3.0, 3.0])
ax.set_ylim([-3.0, 3.0])

ax.tick_params(length=10.0, width=linewidth, labelsize=fontsize)
ax.locator_params(axis='x', nbins=4)
ax.locator_params(axis='y', nbins=4)

ax.spines['bottom'].set_linewidth(linewidth)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['top'].set_linewidth(linewidth)
ax.spines['right'].set_linewidth(linewidth)

ax.set_xlabel('X', fontsize=fontsize, labelpad=10.0)
ax.set_ylabel('Y', rotation=0, va='center_baseline', fontsize=fontsize, labelpad=0.0)

plt.savefig(f'{root_dir}/figs/binary_classification/mean/{prefix_str}_nh_{num_hidden}_seed_{seed}_{rank_str}.pdf')

# Plotting uncertainty
cmap = sns.cubehelix_palette(light=1, as_cmap=True)

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
contour = ax.contourf(grid[0], grid[1], grid_predictions.std(axis=0).reshape(grid_resolution, grid_resolution), cmap=cmap)

ax.scatter(X_predict_0_correct[:, 0], X_predict_0_correct[:, 1], color='tab:blue', label='Class 0 Prediction Correct')
ax.scatter(X_predict_0_incorrect[:, 0], X_predict_0_incorrect[:, 1], color='tab:cyan', label='Class 0 Prediction Incorrect')
ax.scatter(X_predict_1_correct[:, 0], X_predict_1_correct[:, 1], color='tab:red', label='Class 1 Prediction Correct')
ax.scatter(X_predict_1_incorrect[:, 0], X_predict_1_incorrect[:, 1], color='tab:pink', label='Class 1 Prediction Incorrect')

ax.set_xlim([-3.0, 3.0])
ax.set_ylim([-3.0, 3.0])

ax.tick_params(length=10.0, width=linewidth, labelsize=fontsize)
ax.locator_params(axis='x', nbins=4)
ax.locator_params(axis='y', nbins=4)

ax.spines['bottom'].set_linewidth(linewidth)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['top'].set_linewidth(linewidth)
ax.spines['right'].set_linewidth(linewidth)

ax.set_xlabel('X', fontsize=fontsize, labelpad=10.0)
ax.set_ylabel('Y', rotation=0, va='center_baseline', fontsize=fontsize, labelpad=0.0)

plt.savefig(f'{root_dir}/figs/binary_classification/std/{prefix_str}_nh_{num_hidden}_seed_{seed}_{rank_str}.pdf')
