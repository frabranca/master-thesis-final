import torch
from torch.optim import *
from models.model import *
import torch
import h5py
import matplotlib.pyplot as plt
from functions import make_frames_from_events, upload_network
import samna
import numpy as np

# 40a8afe88e8b425da3dee1af2fffce8b
# 14322ea85b894e52a645459acf8eaefc
# id = 'a56eb6d230f14d71b32fe5f1d0a5e5f6'
# f27270d55221461e804cd3de5a4ead95
# id = 'bc44189f205b44798474ac985e8c4c02'
# id = 'df68574de13f41aa9ff9e4551db61a72'
# id = 'c1cdff0d7ecc4f3dbd54e9900330b02b'
# id = '564de547b96641acbd66a8441f8b2b6b'
# id = 'c01c2a2fe3444894be33cbf7d0d824c5'
id = 'd7ca458160924f96ab02bd54a9e7efee'

dynapcnn, config_yml = upload_network(id)
data_path = config_yml['data']['path']
sequence = 'rotation_290'
file = h5py.File(data_path + '/poster_' + str(sequence) + '.h5', 'r')

start = 0
end = 100000
xs = file['events/xs'] #[start:end]
ys = file['events/ys'] #[start:end]
ts = file['events/ts'] #[start:end]
ps = file['events/ps'] #[start:end]

inputs = []

# CONVERT EVENTS TO SAMNA OBJECTS -------------------------------------------------------
# load inputs from h5 file
for i in range(len(xs)):
    speckevent = samna.speck2e.event.Spike()
    speckevent.x = xs[i]
    speckevent.y = ys[i]
    speckevent.feature = ps[i]
    speckevent.timestamp = round((ts[i] - file.attrs['t0'])*1e6)
    speckevent.layer = 0
    inputs.append(speckevent)

print('Events converted to samna objects')

# CHIP TEST ----------------------------------------------------------------------------
outputs = dynapcnn(inputs)

layer_id = 6
outputs_ = [out for out in outputs if out.layer == layer_id]

# frames_input = make_frames_from_events(inputs, (2,90,90))
# frames_ = make_frames_from_events(outputs_, (6,23,23))
# frames_1 = make_frames_from_events(outputs_1, (12,23,23))
# frames_2 = make_frames_from_events(outputs_2, (12,23,23))
# frames_ = make_frames_from_events(outputs_, (16,6,6))
# frames_4 = make_frames_from_events(outputs_4, (32,6,6))
# frames_5 = make_frames_from_events(outputs_5, (16,6,6))
frames_ = make_frames_from_events(outputs_, (15,1,1))

num_outputs_ = [frame.sum().item() for frame in frames_]
max_outputs_ = [frame.max().item() for frame in frames_]

time_ = torch.arange(0, len(frames_)*5, 5)

plt.plot(time_, num_outputs_, label='Output layer' + str(layer_id))
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()

# Save num_outputs_0 locally
torch.save(frames_, 'pool_layer_frames/' + str(sequence) + '_clamped.pth')
# torch.save(torch.tensor([num_outputs_, max_outputs_]), 'layer_0_activity_comparison/clamped_net.pth')
