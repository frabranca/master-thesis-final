# Speck2e Tests

This folder contains the scripts used for the hardware implementation. The following steps were taken:
1- The network weights are quantized according to the constraints of the speck2e (8 bits for weights and 16 bits for membrane potentials).
2- The PyTorch recurrent spiking neural network is converted to a speck-compatible architecture with sinabs modules.
3- The configuration is modified to have concatenation recurrency (here addressed as type B), hence the destinations of the layers are changed manually.
4- An input sequence of events is fed directly into the chip, bypassing the DVS layer and the output spikes are collected and inserted in a final external convolutional layer.
5- This final convolutional layer computes the optical flow vectors, which are compared to the ones obtained in simulation.  

