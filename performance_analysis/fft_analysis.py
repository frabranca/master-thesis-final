import numpy as np
import matplotlib.pyplot as plt

"""
fft_analysis.py

- This script is used to analyse the noise in the network outputs.
- By doing the FFT of the signals, the ratio between useful signal and noise can be calculated.
- It is observed that Integrate-and-Fire (IF) neurons have a higher SNR than Leak-Integrate-and-Fire (LIF).
- Additionally, 2 types of recurrency are tested (here addressed as type A and type B).

"""

gt_signal = np.loadtxt('results_inference/1f5735184f0a47abb20b0409c515b4c1/results/eval_16/2_8/gt_target_vectors.txt')

# RNN-3-A, LIF-3-A
# relu_signal = np.loadtxt('results_inference/0e045bd7fce64ac488134591252a696a/results/eval_23/2_8/gt_pred_vectors.txt')
# if_signal = np.loadtxt('../event_planar/results_inference/31f8f8028de64207aad08ecf3d596897/results/eval_55/2_8/gt_pred_vectors.txt')

# RNN-2-A, IF-2-A
# relu_signal = np.loadtxt('results_inference/1f5735184f0a47abb20b0409c515b4c1/results/eval_11/2_8/gt_pred_vectors.txt')
# if_signal = np.loadtxt('results_inference/1f5735184f0a47abb20b0409c515b4c1/results/eval_12/2_8/gt_pred_vectors.txt')

# RNN-2-B, IF-2-B
relu_signal = np.loadtxt('results_inference/128d8bb90e4e4b8fbc3123471ba0b6ca/results/eval_32/2_8/gt_pred_vectors.txt')
if_signal = np.loadtxt('results_inference/128d8bb90e4e4b8fbc3123471ba0b6ca/results/eval_33/2_8/gt_pred_vectors.txt')

# Calculate the FFT for each column
fft_results_relu = np.fft.fft(relu_signal, axis=0)
fft_results_if = np.fft.fft(if_signal, axis=0)
fft_results_gt = np.fft.fft(gt_signal, axis=0)

# Generate frequency axis
frequency = np.fft.fftfreq(len(if_signal))
frequency_gt = np.fft.fftfreq(len(gt_signal))

# Plot the magnitude spectrum for each column
lim = 0.01
freq = 0.0025
for i in range(8):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    plt.rcParams['font.size'] = 12

    axs[0].plot(frequency_gt, np.abs(fft_results_gt[:, i]), label='GT',  color='g')
    axs[0].set_xlabel('Frequency [Hz]', fontsize=14)
    axs[0].set_ylabel('Magnitude [(pxl/ms)\u00B2]', fontsize=14)
    axs[0].tick_params(axis='both', which='major', labelsize=14)  # Increase fontsize of axis ticks
    # axs[0].set_xlim(-lim, lim)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(frequency, np.abs(fft_results_relu[:, i]), label='RNN-2-S', color='r')
    axs[1].set_xlabel('Frequency (Hz)', fontsize=14)
    axs[1].set_ylabel('Magnitude [(pxl/ms)\u00B2]', fontsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=14)  # Increase fontsize of axis ticks
    # axs[0].set_xlim(-lim, lim)
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(frequency, np.abs(fft_results_if[:, i]), label='IF-2-S', color='b')
    axs[2].set_xlabel('Frequency (Hz)', fontsize=14)
    axs[2].set_ylabel('Magnitude [(pxl/ms)\u00B2]', fontsize=14)
    axs[2].tick_params(axis='both', which='major', labelsize=14)  # Increase fontsize of axis ticks
    # axs[2].set_xlim(-lim, lim)
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'fft_results/fft_comparison_motion_{i+1}.png')
    plt.savefig(f'fft_results/fft_comparison_motion_{i+1}.pdf')
    plt.close(fig)

# Define the range of frequencies for the original signal

power_signal_range = np.where((frequency >= -freq) & (frequency <= freq))
noise_signal_range = np.where((frequency < -freq) | (frequency > freq))

power_signal_range_gt = np.where((frequency_gt >= -freq) & (frequency_gt <= freq))
noise_signal_range_gt = np.where((frequency_gt < -freq) | (frequency_gt > freq))

# Define the power signal
power_gt = (np.abs(fft_results_gt[power_signal_range_gt, :]) ** 2).sum(axis=1)
noise_gt = (np.abs(fft_results_gt[noise_signal_range_gt, :]) ** 2).sum(axis=1)
power_relu = (np.abs(fft_results_relu[power_signal_range, :]) ** 2).sum(axis=1)
noise_relu = (np.abs(fft_results_relu[noise_signal_range, :]) ** 2).sum(axis=1)
power_if = (np.abs(fft_results_if[power_signal_range, :]) ** 2).sum(axis=1)
noise_if = (np.abs(fft_results_if[noise_signal_range, :]) ** 2).sum(axis=1)

snr_avg_gt = (power_gt/noise_gt).mean()
snr_avg_relu = (power_relu/noise_relu).mean()
snr_avg_if = (power_if/noise_if).mean()

print('GT ', 10*np.log10(snr_avg_gt))
print('ReLU', 10*np.log10(snr_avg_relu))
print('IF', 10*np.log10(snr_avg_if))