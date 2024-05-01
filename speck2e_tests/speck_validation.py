import torch
from torch.optim import *
from models.model import *
import torch
import h5py
import matplotlib.pyplot as plt
from functions import create_fake_input_events, make_frames_from_events, upload_network

dynapcnn, config_yml = upload_network("a56eb6d230f14d71b32fe5f1d0a5e5f6")

inputs = create_fake_input_events()    
frames = make_frames_from_events(inputs, (2,90,90))

# CHIP TEST ----------------------------------------------------------------------------

outputs = dynapcnn(inputs)

outputs_0 = [out for out in outputs if out.layer == 0]
frames_0 = make_frames_from_events(outputs_0, (4,45,45))

num_inputs = [frame.sum().item() for frame in frames]
num_outputs_0 = [frame.sum().item() for frame in frames_0]

time_ = torch.arange(0, len(frames)*5, 5)
fig, axs = plt.subplots(2, 1, figsize=(6, 10))
axs[0].plot(time_, num_inputs, label='Input events')
axs[0].set_title('Input events')
plt.grid()

axs[1].plot(time_, num_outputs_0, label='Output layer 0')
axs[1].set_title('Output layer 0')
plt.grid()

plt.tight_layout()
plt.legend()
plt.show()
