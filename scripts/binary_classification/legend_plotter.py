import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp
import jax.random as random

import numpyro
import sys

root_dir = f'/directory/to/quantum_bayesian_neural_networks'
quantum_simulations_dir = f'{root_dir}/scripts/quantum_simulation'

sys.path.insert(1, quantum_simulations_dir)

from quantum_bayesian_neural_network import QBNN

from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
matplotlib.use('Agg')
root_dir =  f'/directory/to/quantum_bayesian_neural_networks'

# create artificial binary classification dataset
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

assert numpyro.__version__.startswith("0.6.0")
numpyro.set_platform("cpu")

n = 2
classical_inference = True
classical_predict = True
isClassifier = True
num_samples = 2000
num_warmup = 1000
num_chains = 1
num_hidden = 5
D_X = 2
num_train_data = 100
num_predict_data = 500

grid_resolution = 100
fontsize = 24
linewidth = 2.0

seed = 0
seed_data = 0
rng_key, rng_key_inference = random.split(random.PRNGKey(seed))
rng_key_grid_predict, rng_key_point_predict = random.split(rng_key)

numpyro.set_host_device_count(num_chains)

X_train, Y_train, X_predict, Y_predict =  get_data(num_train_data=num_train_data, D_X=D_X, num_predict_data=num_predict_data, seed=seed_data)


qbnn = QBNN(n=n, classical_inference=classical_inference, classical_predict=classical_predict, isClassifier=isClassifier, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains, num_hidden=num_hidden)
    
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

for coll in contour.collections:
    coll.set_visible(False)

X_predict_0_correct = X_predict[[pred == 0 and corr == 0 for pred, corr in zip(mean_point_predictions, Y_predict)], :]
X_predict_0_incorrect = X_predict[[pred == 0 and corr == 1 for pred, corr in zip(mean_point_predictions, Y_predict)], :]
X_predict_1_correct = X_predict[[pred == 1 and corr == 1 for pred, corr in zip(mean_point_predictions, Y_predict)], :]
X_predict_1_incorrect = X_predict[[pred == 1 and corr == 0 for pred, corr in zip(mean_point_predictions, Y_predict)], :]

ax.scatter([], [], color="tab:blue", label="Class 0 Prediction Correct")
ax.scatter([], [], color="tab:cyan", label="Class 0 Prediction Incorrect")
ax.scatter([], [], color="tab:red", label="Class 1 Prediction Correct")
ax.scatter([], [], color="tab:pink", label="Class 1 Prediction Incorrect")



ax.axis('off')

axins1 = inset_axes(ax,
                    width="90%",  # width = 50% of parent_bbox width
                    height="10%",  # height : 5%
                    loc='center',
                    bbox_to_anchor=(0.0, -0.275, 1, 1),
                    bbox_transform=ax.transAxes)
cbar = plt.colorbar(contour, orientation="horizontal", cax=axins1, aspect= 10, pad=-4.0)
cbar.ax.spines['bottom'].set_linewidth(linewidth)
cbar.ax.spines['left'].set_linewidth(linewidth)
cbar.ax.spines['top'].set_linewidth(linewidth)
cbar.ax.spines['right'].set_linewidth(linewidth)
cbar.ax.locator_params(axis='x', nbins=8)
cbar.ax.tick_params(length=10.0, width=linewidth, labelsize=fontsize)
cbar.ax.set_title("Posterior Predictive Mean Probability\nof Class Label = 1", fontsize=fontsize, pad=10.0);

ax.legend(loc='center', bbox_to_anchor=(0.5,0.725), fontsize=fontsize, markerscale=3.0, numpoints=1)

plt.savefig(f'{root_dir}/figs/binary_classification/legend_mean.pdf')

# Plotting uncertainty
cmap = sns.cubehelix_palette(light=1, as_cmap=True)

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
contour = ax.contourf(grid[0], grid[1], grid_predictions.std(axis=0).reshape(grid_resolution, grid_resolution), cmap=cmap)
for coll in contour.collections:
    coll.set_visible(False)

ax.scatter([], [], color="tab:blue", label="Class 0 Prediction Correct")
ax.scatter([], [], color="tab:cyan", label="Class 0 Prediction Incorrect")
ax.scatter([], [], color="tab:red", label="Class 1 Prediction Correct")
ax.scatter([], [], color="tab:pink", label="Class 1 Prediction Incorrect")

ax.axis('off')

axins1 = inset_axes(ax,
                    width="90%",  # width = 50% of parent_bbox width
                    height="10%",  # height : 5%
                    loc='center',
                    bbox_to_anchor=(0.0, -0.275, 1, 1),
                    bbox_transform=ax.transAxes)
cbar = plt.colorbar(contour, orientation="horizontal", cax=axins1, aspect= 10, pad=-4.0)
cbar.ax.spines['bottom'].set_linewidth(linewidth)
cbar.ax.spines['left'].set_linewidth(linewidth)
cbar.ax.spines['top'].set_linewidth(linewidth)
cbar.ax.spines['right'].set_linewidth(linewidth)
cbar.ax.locator_params(axis='x', nbins=8)
cbar.ax.tick_params(length=10.0, width=linewidth, labelsize=fontsize)
cbar.ax.set_title("Posterior Predictive Standard Deviation\n(Uncertainty)", fontsize=fontsize, pad=10.0);

ax.legend(loc='center', bbox_to_anchor=(0.5,0.725), fontsize=fontsize, markerscale=3.0, numpoints=1)

plt.savefig(f'{root_dir}/figs/binary_classification/legend_std.pdf')
