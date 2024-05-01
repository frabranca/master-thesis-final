# Master Thesis: Optical Flow Determination using Integrate & Fire Neurons

This repository contains the work that I conducted for my master thesis on spiking neural networks. Some of the scripts is omitted, because they have been forked from a private TU Delft repository (mainly the folders `configs`, `utils`, `models`, `dataloader` and `loss`). 
My project consisted in:

- Preliminary experiments with the Synsense speck device (see [here](https://github.com/frabranca/master-thesis-preliminary)).
- Training of spiking neural networks with ReLU activation functions and converting trained network to spiking using sinabs modules `IAFSqueeze`.
- Testing the converted network and assessing the performance of different architectures (`performance_analysis`).
- Re-training network with additional term in the loss function to minimize the number of synaptic operations (`train_synops.py`, `synops_loss.py`, `synops_analysis.py`).
- Implementations on the speck2e using samna (`speck2e_tests`).


