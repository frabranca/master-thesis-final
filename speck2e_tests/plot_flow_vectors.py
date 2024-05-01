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
import numpy as np

flow_vectors = []

def plot(args, config_parser):
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

    # import pdb; pdb.set_trace()
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
                
                flow = x['flow_vectors'].reshape(-1).cpu()[4:6]
                flow_vectors.append(flow)
            
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

    plot(args, YAMLParser(args.config))                            
    flow_vectors = torch.stack(flow_vectors, dim=1)

    flow1 = flow_vectors[0].numpy()
    flow2 = flow_vectors[1].numpy()

    window_size = 50
    smooth_flow1 = np.convolve(flow1, np.ones(window_size)/window_size, mode='valid')
    smooth_flow2 = np.convolve(flow2, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(15,15))
    plt.title(args.runid)

    # plt.plot(list(range(len(flow1))), np.arctan2(flow2, flow1))
    # plt.plot(list(range(len(smooth_flow1))), np.arctan2(smooth_flow2, smooth_flow1))
    plt.plot(list(range(len(flow1))), flow1)
    plt.plot(list(range(len(flow2))), flow2)

    np.savetxt('flow_vectors_if2.txt', flow_vectors)

    plt.xlabel('Steps')
    plt.ylabel('Flow Components')
    plt.grid()

    directory = 'results_inference/' + args.runid
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)

    plt.savefig(directory + '/flow vectors' + '.png')
    plt.show()
