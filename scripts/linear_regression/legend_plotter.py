import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')
root_dir =  f'/directory/to/quantum_bayesian_neural_networks'

fontsize = 24

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)


ax.plot([], [], 'kx', label='Training Data')

ax.fill_between(
    [], [], [], color='lightblue', label='90% Confidence Interval\nof Predictions'
)

ax.plot([], [], 'blue', ls='solid', lw=2.0, label='Mean Prediciton')

ax.axis('off')

ax.legend(loc='center', bbox_to_anchor=(0.5,0.58), fontsize=fontsize, markerscale=3.0, numpoints=1)

plt.savefig(f'{root_dir}/figs/linear_regression/legend.pdf')
