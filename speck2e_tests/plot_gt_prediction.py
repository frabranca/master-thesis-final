import matplotlib.pyplot as plt
import numpy as np
# Load the data
gt_signal = np.loadtxt('results_inference/1f5735184f0a47abb20b0409c515b4c1/results/eval_16/2_8/gt_target_vectors.txt')

# RNN-2-B, IF-2-B
# relu_signal = np.loadtxt('results_inference/128d8bb90e4e4b8fbc3123471ba0b6ca/results/eval_32/2_8/gt_pred_vectors.txt')
# if_signal = np.loadtxt('results_inference/128d8bb90e4e4b8fbc3123471ba0b6ca/results/eval_33/2_8/gt_pred_vectors.txt')

# RNN-2-A, IF-2-A
relu_signal = np.loadtxt('results_inference/1f5735184f0a47abb20b0409c515b4c1/results/eval_16/2_8/gt_pred_vectors.txt')
if_signal = np.loadtxt('results_inference/1f5735184f0a47abb20b0409c515b4c1/results/eval_12/2_8/gt_pred_vectors.txt')

time = np.arange(0, len(gt_signal)*5, 5)
time_ = np.arange(0, len(relu_signal)*5, 5)
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Plot the first quantity for each file
axs[0].plot(time, gt_signal[:, 0], label='ground truth', color='b')
axs[0].plot(time_, relu_signal[:, 0], label='ReLU prediction', color='r')
axs[0].set_xlabel('Time [ms]', fontsize=16)
axs[0].set_ylabel('TL u [pxl/ms]', fontsize=16)
axs[0].tick_params(axis='both', which='major', labelsize=16)
axs[0].grid(True)
axs[0].legend(loc='best', fontsize=16) 

axs[1].plot(time, gt_signal[:, 0], label='ground truth', color='b')
axs[1].plot(time_, if_signal[:, 0], label='IF prediction', color='r') 
axs[1].set_xlabel('Time [ms]', fontsize=16)
axs[1].set_ylabel('TL u [pxl/ms]', fontsize=16)
axs[1].tick_params(axis='both', which='major', labelsize=16)
axs[1].grid(True)
axs[1].legend(loc='best', fontsize=16) 

# Adjust spacing between subplots
fig.tight_layout()
plt.savefig('pred_gt_compared.pdf')
plt.savefig('pred_gt_compared.png')
# Show the plot
plt.show()
