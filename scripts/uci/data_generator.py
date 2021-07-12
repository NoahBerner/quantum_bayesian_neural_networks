import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp
import jax.random as random

import numpyro
import argparse
import sys

root_dir = f'/directory/to/quantum_bayesian_neural_networks'
quantum_simulations_dir = f'{root_dir}/scripts/quantum_simulation'

sys.path.insert(1, quantum_simulations_dir)
print(sys.path)
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
        
# get artificial regression dataset
def get_data(dataset, split, dtype='float32'):
    dataset_dir = f'{root_dir}/data/uci_data/{dataset}/'
    data = np.loadtxt(f'{dataset_dir}/data.txt').astype(getattr(np, dtype))
    
    index_features = np.loadtxt(f'{dataset_dir}/index_features.txt')
    index_target = np.loadtxt(f'{dataset_dir}/index_target.txt')
    X_unnorm = data[:, index_features.astype(int)]
    Y_unnorm = data[:, index_target.astype(int):index_target.astype(int)+1]
        
    # split into train and test
    index_train = np.loadtxt(f'{dataset_dir}/index_train_{split}.txt').astype(int)
    index_test  = np.loadtxt(f'{dataset_dir}/index_test_{split}.txt').astype(int)
    
    D_X = index_features.size
    D_Y = index_target.size
    
    X_train = X_unnorm[index_train, :]
    Y_train = Y_unnorm[index_train, :]
    X_test = X_unnorm[index_test, :]
    
    # compute normalization constants based on training set
    X_std = np.std(X_train, 0)
    X_std[X_std == 0] = 1. # ensure we don't divide by zero
    X_mean = np.mean(X_train, 0)
    
    Y_std = np.std(Y_train, 0)
    Y_mean = np.mean(Y_train, 0)
    
    # normalize all data
    X_train = (X_train - X_mean)/X_std
    Y_train = (Y_train - Y_mean)/Y_std
    X_test = (X_test - X_mean)/X_std
    
    assert X_train.shape == (index_train.size, D_X)
    assert Y_train.shape == (index_train.size, D_Y)
    assert X_test.shape == (index_test.size, D_X)
    
    return X_train, Y_train, X_test, D_X, D_Y, Y_mean, Y_std

def denormalize_Y(Y, Y_mean, Y_std):
    return Y_std * Y + Y_mean
        

assert numpyro.__version__.startswith('0.6.0')
numpyro.set_platform('cpu') # change this to 'gpu' if available.

parser = argparse.ArgumentParser(description='Parses input variables for uci dataset prediction.')

parser.add_argument('-ci', '--classical_inference', type=str2bool, nargs='?', required=True, help='Use classical inference instead of quantum inference.')
parser.add_argument('-cp', '--classical_predict', type=str2bool, nargs='?', required=True, help='Use classical prediction instead of quantum prediction.')
parser.add_argument('-lri', '--low_rank_initialization', type=str2bool, nargs='?', required=True, help='Use low-rank initialization instead of full-rank initialization.')
parser.add_argument('-n', type=int, nargs='?', required=True, help='Number of qubits used in phase estimation simulation.')
parser.add_argument('--dataset', type=str, nargs='?', required=True, help='Use one of the following: [boston, concrete, energy, kin8nm, naval, power, protein, wine, yacht].')
parser.add_argument('-s', '--split', type=int, nargs='?', required=True, help='Specific split for dataset. Must be in range(20), except for protein -> rnage(5).')
parser.add_argument('--num_hidden', type=int, nargs='?', required=True, help='Number of neurons in the two hidden layers.')
parser.add_argument('--seed', type=int, nargs='?', required=True, help='Seed used for model randomness.')
args = parser.parse_args()

classical_inference = args.classical_inference
classical_predict = args.classical_predict
low_rank_initialization = args.low_rank_initialization
n = args.n
dataset = args.dataset
split = args.split
num_hidden = args.num_hidden
seed = args.seed

print(f'classical_inference: {classical_inference}')
print(f'classical_predict: {classical_predict}')
print(f'low_rank_initialization: {low_rank_initialization}')
print(f'n: {n}')
print(f'dataset: {dataset}')
print(f'split: {split}')
print(f'num_hidden: {num_hidden}')
print(f'seed: {seed}')

datasets = ['boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht']
if dataset not in datasets :
    print(f'Error: Dataset {dataset} is not part of {datasets}.')
    exit(1)
    
if dataset == 'protein':
    if split not in range(5):
        print(f'Error: Split must be between 0 and 4.')
        exit(1)
else :
    if split not in range(20):
        print(f'Error: Split must be between 0 and 19.')
        exit(1)

isClassifier = False
num_samples = 2000
num_warmup = 1000
num_chains = 1
dtype='float32'

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

rng_key_predict, rng_key_inference = random.split(random.PRNGKey(seed))

numpyro.set_host_device_count(num_chains)

X_train, Y_train, X_test, D_X, D_Y, Y_mean, Y_std =  get_data(dataset=dataset, split=split, dtype=dtype)

qbnn = QBNN(n=n, classical_inference=classical_inference, classical_predict=classical_predict, low_rank_initialization=low_rank_initialization, isClassifier=isClassifier, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains, num_hidden=num_hidden)

samples = qbnn.run_inference(rng_key_inference, X_train, Y_train)

predictions = qbnn.predict(rng_key_predict, samples, X_test)
predictions = denormalize_Y(predictions, Y_mean, Y_std)


np.save(f'{root_dir}/data/uci_predictions/{prefix_str}_{dataset}_s_{split}_{rank_str}.npy', predictions)
