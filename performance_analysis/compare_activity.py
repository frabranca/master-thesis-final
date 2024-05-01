import matplotlib.pyplot as plt
import torch

time = torch.arange(0, 1000*5, 5)
time_sinabs = torch.arange(0, 999*5, 5)

sinabs_spikes = torch.load('output_events_sinabs.pth')
speck_spikes = torch.load('output_events_speck.pth')

idx = 253
# Plot the data
plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(time_sinabs[0:idx], sinabs_spikes[0:idx,1],color='b')
plt.plot(time[0:idx], speck_spikes[1][0:idx], color='r')
plt.xlabel('Time [ms]')
plt.ylabel('Number of Spikes Fwd Layer [-]')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time_sinabs[0:idx], sinabs_spikes[0:idx,2], color='b')
plt.plot(time[0:idx], speck_spikes[2][0:idx], color='r')
plt.xlabel('Time [ms]')
plt.ylabel('Number of Spikes Fwd Layer [-]')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time_sinabs[0:idx], sinabs_spikes[0:idx,3], color='b')
plt.plot(time[0:idx], speck_spikes[3][0:idx], color='r')
plt.xlabel('Time [ms]')
plt.ylabel('Number of Spikes Rec Layer [-]')
plt.grid()

plt.show()