import samna
import numpy as np
import torch

def make_layers(dynapcnn, read=[6], num_layers=9):
    assert len(dynapcnn.sequence) <= num_layers
    
    samna_layers = []
    inputs = [dynapcnn.input_shape[1]]

    for i in range(num_layers):
        layer = samna.speck2e.configuration.CnnLayerConfig()
        if i < len(dynapcnn.sequence):

            # padding, kernel, stride
            padding = dynapcnn.sequence[i].conv_layer.padding[0]
            stride = dynapcnn.sequence[i].conv_layer.stride[0]
            kernel = dynapcnn.sequence[i].conv_layer.kernel_size[0]

            layer.dimensions.padding = samna.Vec2_unsigned_char(padding, padding)
            layer.dimensions.stride = samna.Vec2_unsigned_char(stride, stride)
            layer.dimensions.kernel_size = kernel

            # input, output size
            input_size = int(inputs[i])
            output_size = int((inputs[i] - kernel + 2*padding)/ stride + 1)
            inputs.append(output_size)

            layer.dimensions.input_shape.feature_count = dynapcnn.sequence[i].conv_layer.in_channels
            layer.dimensions.input_shape.size = samna.Vec2_unsigned_char(input_size, input_size)
            layer.dimensions.output_shape.feature_count = dynapcnn.sequence[i].conv_layer.out_channels  
            layer.dimensions.output_shape.size = samna.Vec2_unsigned_char(output_size, output_size)

            
            # clamp weights
            dynapcnn.sequence[i].conv_layer.weight.data = torch.clamp(dynapcnn.sequence[i].conv_layer.weight.data, 
                                                                      max=dynapcnn.sequence[i].spk_layer.spike_threshold,
                                                                      min=dynapcnn.sequence[i].spk_layer.min_v_mem)

            # weights, biases
            layer.weights = (dynapcnn.sequence[i].conv_layer.weight.data).int().tolist()

            # resize weights
            # max_desired = dynapcnn.sequence[i].spk_layer.spike_threshold.clone()
            # min_desired = dynapcnn.sequence[i].spk_layer.min_v_mem.clone()
            # max_original = dynapcnn.sequence[i].conv_layer.weight.data.max()
            # min_original = dynapcnn.sequence[i].conv_layer.weight.data.min()
            # resized = (dynapcnn.sequence[i].conv_layer.weight.data.clone() - min_original) / (max_original-min_original)
            # resized = (resized * (max_desired-min_desired) + min_desired).round().detach()

            # layer.weights = (resized).int().tolist()

            if dynapcnn.sequence[i].conv_layer.bias is not None:
                layer.biases = (dynapcnn.sequence[i].conv_layer.bias.data).int().tolist()
            else:
                biases_dimensions = dynapcnn.sequence[i].conv_layer.out_channels
                biases_zeros = np.zeros(biases_dimensions).astype(int).tolist()
                layer.biases = biases_zeros

            # neurons
            neurons_dimensions = (dynapcnn.sequence[i].conv_layer.out_channels, output_size, output_size)
            neurons_values = (np.ones(neurons_dimensions)*1).astype(int).tolist() 
            layer.neurons_initial_value = neurons_values
            
            samna_layers.append(layer)

            # thresholds
            layer.threshold_high = int(dynapcnn.sequence[i].spk_layer.spike_threshold)
            layer.threshold_low = int(dynapcnn.sequence[i].spk_layer.min_v_mem)    
            layer.return_to_zero = False
            
            if i in read:
                layer.monitor_enable = True
            else:
                layer.monitor_enable = False

        else:
            samna_layers.append(layer)

    config = samna.speck2e.configuration.SpeckConfiguration()
    config.cnn_layers = samna_layers
    
    # encoder 1
    # forward gate 1
    config.cnn_layers[0].destinations[0].layer = 1
    config.cnn_layers[0].destinations[0].enable = True

    # forward gate 2
    config.cnn_layers[1].destinations[0].layer = 3
    config.cnn_layers[1].destinations[0].enable = True
    config.cnn_layers[1].destinations[1].layer = 2
    config.cnn_layers[1].destinations[1].enable = True

    # recurrent gate
    config.cnn_layers[2].destinations[0].layer = 1
    config.cnn_layers[2].destinations[0].enable = True
    config.cnn_layers[2].destinations[0].feature_shift = config.cnn_layers[2].dimensions.output_shape.feature_count
    
    # encoder 2 
    # forward gate 1
    config.cnn_layers[3].destinations[0].layer = 4
    config.cnn_layers[3].destinations[0].enable = True

    # forward gate 2 
    config.cnn_layers[4].destinations[0].layer = 6
    config.cnn_layers[4].destinations[0].enable = True
    config.cnn_layers[4].destinations[1].layer = 5
    config.cnn_layers[4].destinations[1].enable = True

    # recurrent gate
    config.cnn_layers[5].destinations[0].layer = 4
    config.cnn_layers[5].destinations[0].enable = True  
    config.cnn_layers[5].destinations[0].feature_shift = config.cnn_layers[5].dimensions.output_shape.feature_count

    # pooling
    config.cnn_layers[6].destinations[0].layer = 12
    config.cnn_layers[6].destinations[0].enable = True

    # config.dvs_layer.monitor_enable = True
    # config.dvs_layer.destinations[0].layer = 0
    # config.dvs_layer.destinations[0].enable = True
    # config.dvs_layer.cut = samna.Vec2_unsigned_char(89, 89)

    # config.readout.enable = True
    # config.readout.output_mode_sel= 0b10
    # config.readout.readout_configuration_sel = 0b11
    # config.readout.readout_pin_monitor_enable = True

    return config