import argparse
import collections
import os

import mlflow
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import EvalCriteria
from models.model import *
from utils.activity_packager import Packager
from utils.homography import HomographyWarping
from utils.iwe import compute_pol_iwe
from utils.utils import load_model, create_model_dir, save_state_dict
from utils.mlflow import log_config, log_results
from utils.visualization import Visualization
import numpy as np
import matplotlib.pyplot as plt

"""
monitor_layers.py

- This script is used to monitor the activity inside the channels of the layers.
- This is used to inspect what features every layer is learning and understand the learning process.

"""

features = []

def read_features_snn(module, input, output):
    input_ = input[0]
    prev_state = input[1]
    z_out, stack = output
    feature = z_out.mean(dim=(2, 3), keepdim=True).view(-1)
    features.append(feature)   

def read_features_ann(module, input, output):
    feature = output.mean(dim=(2, 3), keepdim=True).view(-1)
    features.append(feature)
    
def monitor_layers(args, config_parser, layer):
    mlflow.set_tracking_uri(args.path_mlflow)

    run = mlflow.get_run(args.runid)
    config = config_parser.merge_configs(run.data.params)
    config = config_parser.combine_entries(config)

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs
    config["loader"]["device"] = device

    # data loader
    data = H5Loader(config)
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # model initialization and settings
    num_bins = 2 if config["data"]["voxel"] is None else config["data"]["voxel"]
    model = eval(config["model"]["name"])(config["model"].copy(), config["data"]["crop"], num_bins)
    model = model.to(device)
    model = load_model(args.runid, model, device)
    save_state_dict(args.runid, model)
    model.eval()

    # inference loop
    end_test = False

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
                x = model(inputs["net_input"].to(device))
                read = model.encoder_unet.encoders[layer].conv.register_forward_hook(read_features_ann)
                
            if end_test:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="mlflow model run")
    parser.add_argument(
        "--config",
        default="configs/eval_flow.yml",
        help="config file, overwrites mlflow settings",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument("--path_results", default="results_inference/")
    args = parser.parse_args()

    # launch testing
    layer = 0
    direction = 'left'
    monitor_layers(args, YAMLParser(args.config), layer)                            

    directory = 'results_inference/' + args.runid
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)

    file_path = directory + '/features_layer' + str(layer) + '_' + direction + '.txt'

    features = torch.stack(features, dim=0).cpu()
    features_sum = features.sum(dim=0).numpy()
    np.savetxt(file_path, features_sum, delimiter=',')
