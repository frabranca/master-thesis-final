from sinabs.layers import IAFSqueeze
import torch
from sinabs.activation import MembraneReset, MultiSpike
import matplotlib.pyplot as plt

torch.manual_seed(42)

input_ = torch.rand(1, 5, 5) * 2 - 1
thresh = 0.1
relu = torch.relu(input_)
iaf = IAFSqueeze(batch_size=1, spike_threshold=thresh, min_v_mem=-thresh, spike_fn=MultiSpike, reset_fn=MembraneReset())(input_)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(input_[0], cmap='viridis')
axs[0].set_title('Input')
axs[0].axis('off')

for i in range(input_.shape[1]):
    for j in range(input_.shape[2]):
        axs[0].text(j, i, f'{input_[0, i, j]:.2f}', ha='center', va='center', color='red')

axs[1].imshow(relu[0], cmap='viridis')
axs[1].set_title('ReLU')
axs[1].axis('off')

for i in range(relu.shape[1]):
    for j in range(relu.shape[2]):
        axs[1].text(j, i, f'{relu[0, i, j]:.2f}', ha='center', va='center', color='red')

axs[2].imshow(iaf[0], cmap='viridis')
axs[2].set_title('IAF thresh = ' + str(thresh))
axs[2].axis('off')

for i in range(iaf.shape[1]):
    for j in range(iaf.shape[2]):
        axs[2].text(j, i, f'{iaf[0, i, j]:.2f}', ha='center', va='center', color='red')

plt.show()