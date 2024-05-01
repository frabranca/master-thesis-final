import torch
import samna
from make_config import make_layers
import time
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from sinabs.backend.dynapcnn.io import open_device
from quantize import quantize_network

def upload_network(runid):
    dynapcnn, config_yml = quantize_network(runid, to_upload=True)
    
    def modifier(config):
        new_config = make_layers(dynapcnn)
        return new_config
    
    if len(dynapcnn.sequence)==4: # TYPE A
        dynapcnn.to(device='speck2edevkit', config_modifier=modifier)  
        return dynapcnn, config_yml

    if len(dynapcnn.sequence)==7: # TYPE B
        dynapcnn = upload_typeB(dynapcnn)
        return dynapcnn, config_yml

def create_fake_input_events(events_per_frame=5):
    timewindows = torch.arange(5e3, 5.005e6, 5e3)
    inputs = []

    step = int(timewindows[0]-100/events_per_frame)

    # CREATE FAKE INPUTS FOR VALIDATION -----------------------------------------------------
    for i in range(len(timewindows)):
        t0 = timewindows[i]-timewindows[0]
        if i%100==0:
            # for j in range(events_per_frame):
            event_1 = samna.speck2e.event.Spike(layer=0, feature=0, x=20, y=20, timestamp=t0+step-50)
            event_2 = samna.speck2e.event.Spike(layer=0, feature=0, x=20, y=20, timestamp=t0+step-30)
            event_3 = samna.speck2e.event.Spike(layer=0, feature=0, x=20, y=20, timestamp=t0+step-10)
            event_4 = samna.speck2e.event.Spike(layer=0, feature=0, x=20, y=20, timestamp=t0+step+10)
            event_5 = samna.speck2e.event.Spike(layer=0, feature=0, x=20, y=20, timestamp=t0+step+30)
            event_6 = samna.speck2e.event.Spike(layer=0, feature=0, x=20, y=20, timestamp=t0+step+50)
            inputs.append(event_1)
            inputs.append(event_2)
            inputs.append(event_3)
            inputs.append(event_4)
            inputs.append(event_5)
            inputs.append(event_6)

    print('Events converted to samna objects')
    # inputs = inputs[0:200]
    
    return inputs

def make_frames_from_events(events, dim):
    assert len(dim)==3 # (channels, height, width)
    timewindows = torch.arange(5e3, 5.005e6, 5e3)
    size = (len(timewindows), dim[0], dim[1], dim[2])
    frames = torch.zeros(size)
    timewindows = torch.arange(5e3, 5.005e6, 5e3)
    
    for i, tw in enumerate(timewindows):
        for event in events:
            if isinstance(event, samna.speck2e.event.Spike):
                if event.timestamp <= tw and event.timestamp >= tw-5000:
                    frames[i][event.feature][event.x][event.y] += 1

    return frames

def upload_typeB(dynapcnn):
    # get device
    samna_device = open_device('Speck2eDevKit')
    config = make_layers(dynapcnn)
    samna_device.get_model().apply_configuration(config)
    time.sleep(1)

    # define dynapcnn attributes
    dynapcnn.samna_device = samna_device
    dynapcnn.device = 'speck2edevkit'
    dynapcnn.samna_config = config

    builder = ChipFactory(dynapcnn.device).get_config_builder()
    # Create input source node
    dynapcnn.samna_input_buffer = builder.get_input_buffer()
    # Create output sink node node
    dynapcnn.samna_output_buffer = builder.get_output_buffer()

    # Connect source node to device sink
    dynapcnn.device_input_graph = samna.graph.EventFilterGraph()
    dynapcnn.device_input_graph.sequential(
        [
            dynapcnn.samna_input_buffer,
            dynapcnn.samna_device.get_model().get_sink_node(),
        ]
    )

    # Connect sink node to device
    dynapcnn.device_output_graph = samna.graph.EventFilterGraph()
    dynapcnn.device_output_graph.sequential(
        [
            dynapcnn.samna_device.get_model().get_source_node(),
            dynapcnn.samna_output_buffer,
        ]
    )
    dynapcnn.device_input_graph.start()
    dynapcnn.device_output_graph.start()

    return dynapcnn

