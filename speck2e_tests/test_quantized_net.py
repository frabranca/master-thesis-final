import torch
from torch.optim import *
from models.model import *
import torch
import matplotlib.pyplot as plt
from dataloader.h5 import H5Loader
from quantize import quantize_network
import os

id = 'd7ca458160924f96ab02bd54a9e7efee' # after 10 epochs only
model, config_parser, config = quantize_network(id, to_upload=False)

# data loader
kwargs = config_parser.loader_kwargs
data = H5Loader(config)
dataloader = torch.utils.data.DataLoader(
    data,
    drop_last=True,
    batch_size=config["loader"]["batch_size"],
    collate_fn=data.custom_collate,
    worker_init_fn=config_parser.worker_init_fn,
    **kwargs,
    )

output_events = []
max_spike = []
flow_vectors = []
channels = []

with torch.no_grad():
    while True:
        for inputs in dataloader:
            if data.new_seq:
                data.new_seq = False
                model.reset_states()
            # finish inference loop
            if data.seq_num >= len(data.files):
                end_test = True
                break
            
            # forward pass
            x = model(inputs["net_input"].to('cpu'))
            
            # save flow vectors
            flow_vectors.append(x['flow_vectors'][0][0:2].view(2))

            # save layers output
            input_ = inputs["net_input"].sum()
            out_0_ff1 = model.encoder_unet.encoders[0].recurrent_block.ff1_save.sum().item()
            out_0_ff2 = model.encoder_unet.encoders[0].recurrent_block.ff2_save.sum().item()
            out_0_rec = model.encoder_unet.encoders[0].recurrent_block.rec_save.sum().item()
            out_1_ff1 = model.encoder_unet.encoders[1].recurrent_block.ff1_save.sum().item()
            out_1_ff2 = model.encoder_unet.encoders[1].recurrent_block.ff2_save.sum().item()
            out_1_rec = model.encoder_unet.encoders[1].recurrent_block.rec_save.sum().item()
            out_pool = model.encoder_unet.pooling.pool_save.sum().item()

            output_events.append([input_, out_0_ff1, out_0_ff2, out_0_rec, out_1_ff1, out_1_ff2, out_1_rec, out_pool])
            
            max_input = inputs["net_input"].max()
            max_0_ff1 = model.encoder_unet.encoders[0].recurrent_block.ff1_save.max().item()
            max_0_ff2 = model.encoder_unet.encoders[0].recurrent_block.ff2_save.max().item()
            max_0_rec = model.encoder_unet.encoders[0].recurrent_block.rec_save.max().item()
            max_1_ff1 = model.encoder_unet.encoders[1].recurrent_block.ff1_save.max().item()
            max_1_ff2 = model.encoder_unet.encoders[1].recurrent_block.ff2_save.max().item()
            max_1_rec = model.encoder_unet.encoders[1].recurrent_block.rec_save.max().item()
            max_pool = model.encoder_unet.pooling.pool_save.max().item()

            max_spike.append([max_input, max_0_ff1, max_0_ff2, max_0_rec, max_1_ff1, max_1_ff2, max_1_rec, max_pool])

        if end_test:
            break

f=20
down = 50
lim = 300
time = torch.arange(0, len(output_events)*5, 5)/1000
time_ = torch.arange(0, len(output_events)*5+5, 5)/1000
flow_vectors = torch.stack(flow_vectors)*1000
# plt.figure(figsize=(12,6))

# Plot 1: Input Events, Encoder 1 Output, Encoder 2 Output
speck_events = torch.load('layer_0_activity_comparison/not_clamped.pth')
speck_events_clamped = torch.load('layer_0_activity_comparison/clamped_nettt.pth')
simulation_events = torch.tensor(output_events)

max_spike = torch.tensor(max_spike)
# plt.subplot(2,1,1)
# plt.plot(time[down:lim], simulation_events[down:lim,1], 'mediumseagreen', linewidth=3.0)
# plt.plot(time_[down:lim], speck_events[0,down:lim], 'royalblue', linewidth=3.0)
# plt.plot(time_[down:lim], speck_events_clamped[0,down:lim], 'orangered', linewidth=3.0)
# plt.xlabel('Time [s]', fontsize=f)
# plt.ylabel('Number of Events [-]', fontsize=f)
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)
# plt.grid()
# plt.legend(['simulation', 'speck2e', 'speck2e clamped'], fontsize=f)
# plt.tight_layout()

# plt.subplot(2,1,2)
# plt.plot(time[0:lim], max_spike[0:lim,1], 'orangered', linewidth=2.0)
# plt.plot(time_[0:lim], samna_events[1,0:lim], 'royalblue', linewidth=2.0)
# plt.xlabel('Time [ms]', fontsize=f)
# plt.ylabel('Max Number of Spikes [-]', fontsize=f)
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)
# plt.grid()
# plt.legend(['simulation', 'speck2e'], fontsize=f)

# plt.legend(['Input Events', 'Encoder 0 ff1', 'Encoder 0 ff2', 'Encoder 0 rec', 'Encoder 1 ff1', 'Encoder 1 ff2', 'Encoder 1 rec', 'Pooling'])    

file_ = '/home/francesco/event_planar_fra/pool_layer_frames/rotation_090.pth'
file_clamped = '/home/francesco/event_planar_fra/pool_layer_frames/rotation_290_clamped.pth'

# Load the file
data_ = torch.load(file_)
data_clamped = torch.load(file_clamped)
model, config_parser, config = quantize_network(id, to_upload=False)

flow_ = model.encoder_unet.pred.conv2d(data_).squeeze().detach().cpu().numpy()*1000
flow_clamped = model.encoder_unet.pred.conv2d(data_clamped).squeeze().detach().cpu().numpy()*1000

plt.figure(figsize=(10,6))
# plt.subplot(211)
# plt.plot(time, flow_vectors[:,0], color='mediumseagreen', linewidth=2.0)
# plt.plot(time_, -flow_[:,3], color='royalblue', linewidth=2.0)
# plt.ylabel('u [pxl/s]', fontsize=f)
# plt.xlabel('Time [s]', fontsize=f)
# plt.grid()
# plt.legend(['simulation', 'speck2e'], loc='upper right', fontsize=f)    
# plt.xticks(fontsize=f)
# plt.yticks(fontsize=f)
# plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=True)

# plt.subplot(212)
plt.plot(time, flow_vectors[:,0], color='mediumseagreen', linewidth=2.0)
plt.plot(time_, -flow_clamped[:,3], color='orangered', linewidth=2.0)
plt.xlabel('Time [s]', fontsize=f)
plt.ylabel('u [pxl/s]', fontsize=f)
plt.grid()
plt.legend(['simulation', 'speck2e clamped'], loc='upper right', fontsize=f)    
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)

plt.tight_layout()

# Create directory if it doesn't exist
directory = 'results_inference/' + id
if not os.path.exists(directory):
    os.makedirs(directory)
    
plt.savefig('results_inference/' + id + '/spike_activity_comparison.pdf')
plt.show()
# torch.save(torch.tensor(output_events), 'results_inference/' + id + '/output_events_sinabs.pth')
