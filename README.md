# Master Thesis: Optical Flow Determination using Integrate & Fire Neurons
**DISCALIMER**: some of the scripts are omitted, because they have been forked from a private TU Delft repository (mainly the folders `configs`, `utils`, `models`, `dataloader` and `loss`). 
For this reason, this repository cannot be used directly but serves to demonstrate the work that has been carried out.

For additional details on the project you can read the thesis [here](https://github.com/frabranca/master-thesis-final/blob/master/Optical_Flow_Determination_using_Neuromorphic_Hardware_with_Integrate_and_Fire_Neurons.pdf).
My project consisted in:

- Preliminary experiments with the Synsense speck device (see [here](https://github.com/frabranca/master-thesis-preliminary)).
- Training of spiking neural networks with ReLU activation functions and converting trained network to spiking using [sinabs](https://sinabs.readthedocs.io/en/v2.0.0/) modules `IAFSqueeze`.
- Testing the converted network and assessing the performance of different architectures (`performance_analysis/`).
- Re-training network with additional term in the loss function to minimize the number of synaptic operations (`train_synops.py`, `synops_loss.py`, `synops_analysis.py`).
- Optimizing other hyperparameters to minimize synaptic operations (number of layers, channels, stride etc.).
- Implementation on the speck2e using [samna](https://synsense-sys-int.gitlab.io/samna/) (`speck2e_tests/`).


