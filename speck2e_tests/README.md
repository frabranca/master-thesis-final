# Speck2e Tests

This folder contains the scripts used for the hardware implementation. The following steps were taken:

1. The network weights are **_quantized_** according to the constraints of the speck2e (8 bits for weights and 16 bits for membrane potentials).
2. The PyTorch recurrent spiking neural network is converted to a **_speck-compatible_** architecture with sinabs modules.
3. The configuration is modified to have **_concatenation_** recurrency (here addressed as type B, see schematic [here](https://github.com/frabranca/master-thesis-final/blob/master/speck2e_tests/recurrency_block_diagram.png)). At the time of this project, the recurrent connections were not directly supported by samna, hence the destinations of the layers are changed manually.
4. An input sequence of events is fed directly into the chip, bypassing the DVS layer and the output spikes are collected outisde of the chip and inserted in a final external convolutional layer.
5. This final convolutional layer computes the optical flow vectors, which are compared to the ones obtained in simulation.  


