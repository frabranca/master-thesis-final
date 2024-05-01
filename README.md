# Loihi lander - Vision pipeline

**WORK IN PROGRESS**

## Installation

This project uses Python >= 3.7.3 and we strongly recommend the use of virtual environments. If you don't have an environment manager yet, we recommend `pyenv`. It can be installed via:

```
curl https://pyenv.run | bash
```

Make sure your `~/.bashrc` file contains the following:

```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

After that, restart your terminal and run:

```
pyenv update
```

To set up your environment with `pyenv` first install the required python distribution and make sure the installation is successful (i.e., no errors nor warnings):

```
pyenv install -v 3.7.3
```

Once this is done, set up the environment and install the required libraries:

```
pyenv virtualenv 3.7.3 event_flow
pyenv activate event_flow

pip install --upgrade pip==20.0.2

cd event_planar/
```

To install our dependencies:

```
pip install -r requirements.txt
```

Code is formatted with Black (PEP8) using a pre-commit hook. To configure it, run:

```
pre-commit install
```

### Download datasets

In this work, we use the our own event camera dataset recorded at TU Delft's CyberZoo research and test laboratory. The dataset can be downloaded in the expected HDF5 data format from [here](https://surfdrive.surf.nl/files/index.php/s/AKTNpOvQ5mUTjSd) (Download size: 5.1 GB. Uncompressed size: 5.8 GB.
), and is expected at `../datasets/`.


Details about the structure of these files can be found in `tools/data/`.

### Download models

Most of our pretrained models are not available yet. Ask Fede and he will send you the latest.

Benchmarking models can be found [here](https://surfdrive.surf.nl/files/index.php/s/tnmbWovh77IzY74).

In this project we use [MLflow](https://www.mlflow.org/docs/latest/index.html#) to keep track of the experiments. To visualize the models that are available, alongside other useful details and evaluation metrics, run the following from the home directory of the project:

```
mlflow ui
```

and access [http://127.0.0.1:5000](http://127.0.0.1:5000) from your browser of choice.

## Inference

To estimate planar optical flow from the test sequences from our dataset and compare against ground-truth data, run:

```
python eval_flow.py <model_runid>

# for example:
python eval_flow.py a444f7306185412586ecd3826bff7a3e
```

where `<model_runid>` is the ID of MLflow run to be evaluated (check MlFlow).

## Training

Run:

```
python train_flow.py
```

to train a Loihi-compatible spiking neural network. In `configs/`, you can find configuration files and vary the training settings (e.g., model, number of input events, activate/deactivate visualization). For other models available, see `models/model.py`. 

During and after the training, information about your run can be visualized through MLflow.

## Validate Loihi implementation

First of all, you woud need the PyTorch model of the vision and control networks that were used to collect the `.bag` file that you want to validate. Models are currently not available. Ask Fede and he will send you the latest.

Once you have the file and models, initialize submodules:

```
git submodule update --init --recursive
```

and install [loihi-lander](https://github.com/Huizerd/loihi-lander) as Python package via:

```
python setup.py install
```

Once this is done, adjust `config/eval_flow.yml` or create your own config file and run:

```
python eval_online.py <vision_model_runid> <control_model_runid>

# for example:
python eval_online.py vision_last single-axis-discrete-float-controller-lifspenc
```

## Uninstalling pyenv

Once you finish using our code, you can uninstall `pyenv` from your system by:

1. Removing the `pyenv` configuration lines from your `~/.bashrc`.
2. Removing its root directory. This will delete all Python versions that were installed under the `$HOME/.pyenv/versions/` directory:

```
rm -rf $HOME/.pyenv/
```
