# Master Thesis: Optical Flow Determination using Integrate & Fire Neurons
**DISCALIMER**: some of the scripts are omitted, because they have been forked from a private TU Delft repository (mainly the folders `configs`, `utils`, `models`, `dataloader` and `loss`). 
For this reason, this repository cannot be used directly but serves to demonstrate the work that has been carried out.

## Abstract

Spiking neural networks implemented for sensing and control of robots have the potential to achieve lower latency and power consumption by processing information sparsely and
asynchronously. They have been used on neuromorphic devices to estimate optical flow for micro air vehicles navigation, however robotic implementations have been limited to hardware setups
with sensing and processing as separate systems. This article investigates a new approach for training a spiking neural network for optical flow to be deployed on the speck2e device from
Synsense. The method takes into account the restrictions of the speck2e in terms of network architecture, neuron model, and number of synaptic operations and it involves training a
recurrent neural network with ReLU activation functions, which is subsequently converted into a spiking network. A system of weight rescaling is applied after conversion, to ensure optimal
information flow between the layers. Our study shows that it is possible to estimate optical flow with Integrate-and-Fire neurons. However, currently, the optical flow estimation performance is
still hampered by the number of synaptic operations. As a result, the network presented in this work is able to estimate optical flow in a range of [-4, 1] pixel/s.

For additional details on the project you can read the thesis [here](https://github.com/frabranca/master-thesis-final/blob/master/Optical_Flow_Determination_using_Neuromorphic_Hardware_with_Integrate_and_Fire_Neurons.pdf).

## Project Steps

- Preliminary experiments with the Synsense speck device (see [here](https://github.com/frabranca/master-thesis-preliminary)).
- Design of a recurrent mechanism based on _concatention_ to ensure a short-term memory system inside the network and perform _integration in time_.
- Training of SNNs with ReLU activation functions and converting trained model to spiking using [sinabs](https://sinabs.readthedocs.io/en/v2.0.0/) modules `IAFSqueeze`.
- Testing the converted network and assessing the performance of different architectures (`performance_analysis/`).
- Re-training network with additional term in the loss function to minimize the number of _synaptic operations_ (`train_synops.py`, `synops_loss.py`, `synops_analysis.py`).
- Optimizing other _hyperparameters_ to minimize synaptic operations, such as number of layers, channels and stride.
- Implementation on the speck2e using [samna](https://synsense-sys-int.gitlab.io/samna/) (`speck2e_tests/`).

## Project Outcomes
- This project served to explore the limitations and challenges of training a speck2e-compatible IF network to estimate optical flow. 
- Starting from this analysis it might be possible to further improve the network, by re-training it under the constraints of the device.
- One limitation that is still constraining the speck2e is the limit on synaptic operations per second.
- This represent a complex challenge, considering that all the information has to be sent through the spikes between layers, since there is no leak system in the device.


