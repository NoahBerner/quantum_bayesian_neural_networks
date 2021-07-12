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
        
def get_data(num_train_data, D_X, sigma_obs, num_predict_data, seed):
    D_Y = 1  # create 1d outputs
    np.random.seed(seed)
    X_train = jnp.linspace(-1, 1, num_train_data)
    X_train = jnp.power(X_train[:, np.newaxis], jnp.arange(D_X))
    W = 0.5 * np.random.randn(D_X)
    Y_train = jnp.dot(X_train, W) + 0.5 * jnp.power(0.5 + X_train[:, 1], 2.0) * jnp.sin(4.0 * X_train[:, 1])
    Y_train += sigma_obs * np.random.randn(num_train_data)
    Y_train = Y_train[:, np.newaxis]
    Y_train -= jnp.mean(Y_train)
    Y_train /= jnp.std(Y_train)

    assert X_train.shape == (num_train_data, D_X)
    assert Y_train.shape == (num_train_data, D_Y)

    X_predict = jnp.linspace(-1.3, 1.3, num_predict_data)
    X_predict = jnp.power(X_predict[:, np.newaxis], jnp.arange(D_X))

    assert X_predict.shape == (num_predict_data, D_X)
    
    return X_train, Y_train, X_predict


assert numpyro.__version__.startswith('0.6.0')
numpyro.set_platform('cpu')  # change this to 'gpu' if available.

parser = argparse.ArgumentParser(description='Parses input variables for linear regression task on quantum BNN.')

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

isClassifier = False
num_samples = 2000
num_warmup = 1000
num_chains = 1
D_X = 3
num_train_data = 100
num_predict_data = 500
sigma_obs = 0.05

fontsize = 25
linewidth = 2.0

seed = args.seed
seed_data = args.seed
rng_key_predict, rng_key_inference = random.split(random.PRNGKey(seed))

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

X_train, Y_train, X_predict =  get_data(num_train_data=num_train_data, D_X=D_X, sigma_obs=sigma_obs, num_predict_data=num_predict_data, seed=seed_data)


qbnn = QBNN(n=n, classical_inference=classical_inference, classical_predict=classical_predict, low_rank_initialization=low_rank_initialization, isClassifier=isClassifier, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains, num_hidden=num_hidden)

samples = qbnn.run_inference(rng_key_inference, X_train, Y_train)

predictions =  qbnn.predict(rng_key_predict, samples, X_predict)

# compute mean prediction and confidence interval around median
mean_prediction = jnp.mean(predictions, axis=0)
percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)

# make plots
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

# plot training data
ax.plot(X_train[:, 1], Y_train[:, 0], 'kx', label='Training Data')
# plot 90% confidence level of predictions
ax.fill_between(
    X_predict[:, 1], percentiles[0, :], percentiles[1, :], color='lightblue', label='90% Confidence Interval of Predictions'
)
# plot mean prediction
ax.plot(X_predict[:, 1], mean_prediction, 'blue', ls='solid', lw=2.0, label='Mean Prediciton')

ax.set_xlim([-1.3, 1.3])
ax.set_ylim([-4.5, 4.5])

ax.tick_params(length=10.0, width=linewidth, labelsize=fontsize)
ax.locator_params(axis='x', nbins=8)
ax.locator_params(axis='y', nbins=8)

ax.spines['bottom'].set_linewidth(linewidth)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['top'].set_linewidth(linewidth)
ax.spines['right'].set_linewidth(linewidth)

#ax.legend()
ax.set_xlabel('X', fontsize=fontsize, labelpad=10.0)
ax.set_ylabel('Y', rotation=0, va='center_baseline', fontsize=fontsize, labelpad=0.0)


plt.savefig(f'{root_dir}/figs/linear_regression/{prefix_str}_nh_{num_hidden}_seed_{seed}_{rank_str}.pdf')

