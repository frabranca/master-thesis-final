import matplotlib.pyplot as plt
import numpy as np

# Define the rate function
def if_function(t):
    return np.array(t/0.1, dtype=int)

def relu_function(t):
        return np.array(t)

# Generate events based on the rate function
time_points = np.linspace(0, 1.0, 300)  # Adjust the time range as needed
events_relu = relu_function(time_points)
events_if = if_function(time_points)

events_relu = np.random.rand(len(time_points)) < events_relu / max(events_relu)
f = 16
# Plot the event plot
plt.figure(figsize=(8, 6))

# Plot the first subplot
plt.subplot(2, 2, 1)
plt.plot(time_points, time_points, color='royalblue')
plt.xlabel('x [-]', fontsize=f)
plt.ylabel('ReLU(x) [-]', fontsize=f)
plt.xticks(np.arange(0,1.2,0.2), fontsize=f)
plt.yticks(np.arange(0,1.2,0.2), fontsize=f)
plt.grid()

# Plot the second subplot
plt.subplot(2, 2, 2)
plt.plot(time_points, if_function(time_points), color='royalblue')
plt.xlabel('x [-]', fontsize=f)
plt.ylabel('IAFSqueeze(x) [-]', fontsize=f)
plt.xticks(np.arange(0,1.2,0.2), fontsize=f)
plt.yticks(np.arange(0,12,2), fontsize=f)
plt.grid()

# Plot the fourth subplot
plt.subplot(2, 1, 2)
plt.eventplot(time_points[events_relu], lineoffsets=0.5, linelengths=1.0, color='royalblue')
plt.xlabel('Time Steps [-]', fontsize=f)
plt.ylabel('Events [-]', fontsize=f)
plt.yticks(np.arange(0,2,1), fontsize=f)
plt.xticks(np.arange(0,1.2,0.2), fontsize=f)

plt.tight_layout()
plt.savefig('rate_plot.pdf')
plt.show()
