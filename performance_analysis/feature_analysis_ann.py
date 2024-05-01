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
import matplotlib.pyplot as plt
import time 
import torch
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
import time 
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sinabs import SNNAnalyzer

layer = 0
shapes = [(4,64,64), (8,32,32), (15,16,16)]
features = []
synops = []
def read_spikes0(module, input, output):
    # out = (output[]/ 0.001).view(-1)
    cmap = 'hot'
    # plt.plot(torch.arange(0,15), out.cpu(), 'b')
    # plt.subplot(1,3,1)
    # plt.imshow(out[0].cpu(), cmap=cmap)
    # plt.subplot(1,3,2)
    # plt.imshow(out[1].cpu(), cmap=cmap)
    # plt.subplot(1,3,3)
    # plt.imshow(out[2].cpu(), cmap=cmap)
    # plt.colorbar()

    plt.pause(0.001)  # Pause to allow the plot to update

def feature_analysis(args, config_parser):
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

    # Create the plot with a specific figure size
    fig, ax = plt.subplots(figsize=(7, 7))
    # img = ax.imshow(torch.zeros(shapes[layer][1], shapes[layer][2]).cpu())

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
                # print(analyzer.get_layer_statistics()['parameter'])
                
                # import pdb; pdb.set_trace()
                # read_0 = model.encoder_unet.encoders[layer].recurrent_block.register_forward_hook(read_spikes0)
                read_0 = model.encoder_unet.pooling.register_forward_hook(read_spikes0)
                
            if end_test:
                break

    plt.show()  # Show the final plot

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
    feature_analysis(args, YAMLParser(args.config))
    print(synops)