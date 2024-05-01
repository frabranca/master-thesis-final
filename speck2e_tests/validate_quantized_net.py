import torch
from torch.optim import *
from models.model import *
import torch
import matplotlib.pyplot as plt
from quantize import quantize_network
from functions import create_fake_input_events, make_frames_from_events

# 14322ea85b894e52a645459acf8eaefc
# a56eb6d230f14d71b32fe5f1d0a5e5f6
model_quantized, config_parser, config = quantize_network("a56eb6d230f14d71b32fe5f1d0a5e5f6", to_upload=False)

# data loader
kwargs = config_parser.loader_kwargs

events = []
max_spike = []
flow_vectors = []

visualize_features = False
input_events = create_fake_input_events()
dataloader = make_frames_from_events(input_events, (2,90,90))

outputs_0 = []  # Create an empty list to store the outputs

with torch.no_grad():
    for frame_num, inputs in enumerate(dataloader):
        x = model_quantized(inputs.unsqueeze(dim=0).to('cpu'))

        # verify only first layer of first encoder
        sum_input = inputs.sum().item()    
        sum0 = model_quantized.states[0].sum().item()
        max0 = model_quantized.states[0].max().item()
        events.append([sum_input, sum0])
        max_spike.append([max0])  

        # if visualize_features and sum1 > 1:
        #     plt.figure(figsize=(20, 10))
        #     for i in range(8):
        #         plt.subplot(2, 8, i+1)
        #         plt.imshow(model_quantized.states[1].squeeze(dim=0)[i], cmap='hot')
        #         plt.xlabel('Time [ms]')
        #         plt.ylabel('Potential')
        #         plt.title(f'Feature {i+1} '+str(frame_num))
        #         plt.subplot(2, 8, i+9)
        #         plt.imshow(model_quantized.encoder_unet.encoders[1].recurrent_block.activation_rec.v_mem.squeeze(dim=0)[i], cmap='hot')
        #         plt.xlabel('Time [ms]')
        #         plt.ylabel('Potential')
        #         plt.title(f'Feature {i+1}')
        #     plt.tight_layout()
        #     plt.show()

time = torch.arange(0, len(events)*5, 5)
events = torch.tensor(events)

fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(time, max_spike)
ax1.set_xlabel('Time [ms]')
ax1.set_ylabel('Maximum Number of Spikes per Neuron [-]')
ax1.grid()
ax1.legend(['Encoder 1 Max Spikes', 'Encoder 2 Max Spikes'])

fig2, axs2 = plt.subplots(2, 1, figsize=(15, 10))
axs2[0].plot(time, events[:,0])
axs2[0].set_xlabel('Time [ms]')
axs2[0].set_ylabel('Input Events [-]')
axs2[0].grid()

axs2[1].plot(time, events[:,1])
axs2[1].set_xlabel('Time [ms]')
axs2[1].set_ylabel('Output Events [-]')
axs2[1].grid()

plt.tight_layout()
plt.show()

print()
