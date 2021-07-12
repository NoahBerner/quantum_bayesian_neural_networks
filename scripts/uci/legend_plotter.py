import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

matplotlib.use('Agg')
root_dir =  f'/directory/to/quantum_bayesian_neural_networks'

fontsize = 28
linewidth = 3.9

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

ax.plot([], [], c='black', ls='dashed', lw=linewidth, label='Classical Prediction')
ax.plot([], [], c='black', ls='solid', lw=linewidth, label='Quantum Prediction')

ax.plot([], [], c='tab:olive', ls='dotted', lw=linewidth, label='5 Neurons per Hidden Layer')
ax.plot([], [], c='tab:cyan', ls='dotted', lw=linewidth, label='10 Neurons per Hidden Layer')
ax.plot([], [], c='tab:orange', ls='dotted', lw=linewidth, label='15 Neurons per Hidden Layer')
ax.plot([], [], c='tab:blue', ls='dotted', lw=linewidth, label='20 Neurons per Hidden Layer')

ax.axis('off')

ax.legend(loc='center', bbox_to_anchor=(0.5,0.5), fontsize=fontsize, markerscale=3.0, numpoints=1)

plt.savefig(f'{root_dir}/figs/uci/legend.pdf')
