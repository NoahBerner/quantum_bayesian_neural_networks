import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

root_dir =  f'/directory/to/quantum_bayesian_neural_networks'

matplotlib.use('Agg')
        
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
    Y_test = Y_unnorm[index_test, :]
    
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
    Y_test = (Y_test - Y_mean)/Y_std
    
    assert X_train.shape == (index_train.size, D_X)
    assert Y_train.shape == (index_train.size, D_Y)
    assert X_test.shape == (index_test.size, D_X)
    assert Y_test.shape == (index_test.size, D_Y)
    
    return X_train, Y_train, X_test, Y_test, D_X, D_Y, Y_mean, Y_std
    
def normalize_Y(Y, Y_mean, Y_std):
    return (Y - Y_mean)/Y_std
        
def log_likelihood(predictions, targets):
    prediction_mean = predictions.mean(axis=0)
    prediction_std = predictions.std(axis=0)
    targets = targets.reshape((targets.shape[0],))

    return scipy.stats.norm.logpdf(targets, prediction_mean, prediction_std).mean()

dataset = 'wine'
ns = [5, 7, 9, 11, 13]
num_hiddens = [5, 10, 15, 20]
splits = list(range(20))
seed = 0

saved_predictions_dir = f'{root_dir}/data/uci_predictions'

fontsize = 25
linewidth = 2.0

cicp_log_likelihood_means = np.empty(len(num_hiddens))
cicp_log_likelihood_errors = np.empty(len(num_hiddens))
ciqp_log_likelihood_means = np.empty((len(num_hiddens), len(ns)))
ciqp_log_likelihood_errors = np.empty((len(num_hiddens), len(ns)))
for k, num_hidden in enumerate(num_hiddens):
    cicp_log_likelihoods = np.empty(len(splits))
    for i, split in enumerate(splits) :
        _, _, _, Y_test, _, _, Y_mean, Y_std =  get_data(dataset=dataset, split=split)
        cicp_filename = f'cicp_{dataset}_s_{split}_nh_{num_hidden}_seed_{seed}_fr.npy'
        cicp_predictions = np.load(f'{saved_predictions_dir}/{cicp_filename}')
        normalized_cicp_predictions = normalize_Y(cicp_predictions, Y_mean, Y_std)
        
        cicp_log_likelihood = log_likelihood(normalized_cicp_predictions, Y_test)
        cicp_log_likelihoods[i] = cicp_log_likelihood
    
    ciqp_log_likelihoods = np.empty((len(ns), len(splits)))
    for i, n in enumerate(ns) :
        for j, split in enumerate(splits) :
            _, _, _, Y_test, _, _, Y_mean, Y_std =  get_data(dataset=dataset, split=split)
        
            ciqp_filename = f'ciqp_n_{n}_{dataset}_s_{split}_nh_{num_hidden}_seed_{seed}_fr.npy'
            ciqp_predictions = np.load(f'{saved_predictions_dir}/{ciqp_filename}')
            normalized_ciqp_predictions = normalize_Y(ciqp_predictions, Y_mean, Y_std)
            
            ciqp_log_likelihood = log_likelihood(normalized_ciqp_predictions, Y_test)
            ciqp_log_likelihoods[i, j] = ciqp_log_likelihood
            
    cicp_log_likelihood_mean = cicp_log_likelihoods.mean(axis=0)
    cicp_log_likelihood_std = cicp_log_likelihoods.std(axis=0)
    cicp_log_likelihood_means[k] = cicp_log_likelihood_mean
    cicp_log_likelihood_errors[k] = cicp_log_likelihood_std/np.sqrt(len(splits))
    
    
    ciqp_log_likelihood_mean = ciqp_log_likelihoods.mean(axis=1)
    ciqp_log_likelihood_std = ciqp_log_likelihoods.std(axis=1)
    ciqp_log_likelihood_means[k, :] = ciqp_log_likelihood_mean
    ciqp_log_likelihood_errors[k, :] = ciqp_log_likelihood_std/np.sqrt(len(splits))
    
    
print('cicp_log_likelihood_mean: ' + str(cicp_log_likelihood_mean))
print('cicp_log_likelihood_errors: ' + str(cicp_log_likelihood_errors))
print('ciqp_log_likelihood_means: ' + str(ciqp_log_likelihood_means))
print('ciqp_log_likelihood_errors: ' + str(ciqp_log_likelihood_errors))

# make plots
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

cicp_error_x = 4.75
# plot data

ax.axhline(cicp_log_likelihood_means[0], c='tab:olive', ls='dashed', lw=1.0)
cicp_nh_5_errorbar = ax.errorbar([cicp_error_x], cicp_log_likelihood_means[0], yerr=cicp_log_likelihood_errors[0], c='tab:olive', ls='dashed', lw=1.0, elinewidth=1.0, capsize=5.0)
cicp_nh_5_errorbar[-1][0].set_linestyle('--')
ax.errorbar(ns, ciqp_log_likelihood_means[0,:], yerr=ciqp_log_likelihood_errors[0,:], c='tab:olive', ls='solid', lw=2.0, elinewidth=1.0, capsize=5.0)

ax.axhline(cicp_log_likelihood_means[1], c='tab:cyan', ls='dashed', lw=1.0)
cicp_nh_10_errorbar = ax.errorbar([cicp_error_x], cicp_log_likelihood_means[1], yerr=cicp_log_likelihood_errors[1], c='tab:cyan', ls='dashed', lw=1.0, elinewidth=1.0, capsize=5.0)
cicp_nh_10_errorbar[-1][0].set_linestyle('--')
ax.errorbar(ns, ciqp_log_likelihood_means[1,:], yerr=ciqp_log_likelihood_errors[1,:], c='tab:cyan', ls='solid', lw=2.0, elinewidth=1.0, capsize=5.0)

ax.axhline(cicp_log_likelihood_means[2], c='tab:orange', ls='dashed', lw=1.0)
cicp_nh_15_errorbar = ax.errorbar([cicp_error_x], cicp_log_likelihood_means[2], yerr=cicp_log_likelihood_errors[2], c='tab:orange', ls='dashed', lw=1.0, elinewidth=1.0, capsize=5.0)
cicp_nh_15_errorbar[-1][0].set_linestyle('--')
ax.errorbar(ns, ciqp_log_likelihood_means[2,:], yerr=ciqp_log_likelihood_errors[2,:], c='tab:orange', ls='solid', lw=2.0, elinewidth=1.0, capsize=5.0)

ax.axhline(cicp_log_likelihood_means[3], c='tab:blue', ls='dashed', lw=1.0)
cicp_nh_20_errorbar = ax.errorbar([cicp_error_x], cicp_log_likelihood_means[3], yerr=cicp_log_likelihood_errors[3], c='tab:blue', ls='dashed', lw=1.0, elinewidth=1.0, capsize=5.0)
cicp_nh_20_errorbar[-1][0].set_linestyle('--')
ax.errorbar(ns, ciqp_log_likelihood_means[3,:], yerr=ciqp_log_likelihood_errors[3,:], c='tab:blue', ls='solid', lw=2.0, elinewidth=1.0, capsize=5.0)
        
ax.set_xlim([4.5, 13.5])
#ax.set_ylim([1.0, 1.55])

ax.tick_params(length=10.0, width=linewidth, labelsize=fontsize)
ax.set_xticks(ns)
ax.set_xticklabels(ns)
#ax.locator_params(axis='x', nbins=10)
ax.locator_params(axis='y', nbins=6)

ax.spines['bottom'].set_linewidth(linewidth)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['top'].set_linewidth(linewidth)
ax.spines['right'].set_linewidth(linewidth)

#ax.legend()
ax.set_xlabel('n', fontsize=fontsize, labelpad=10.0)
ax.set_ylabel('Log Likelihood', rotation=90, va='center_baseline', fontsize=fontsize, labelpad=20.0)

plt.savefig(f'{root_dir}/figs/uci/log_likelihood_{dataset}.pdf')
