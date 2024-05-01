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
from sinabs_network import SpeckNet
import numpy as np

flow_vectors = []

def test(args, config_parser):
    mlflow.set_tracking_uri(args.path_mlflow)

    run = mlflow.get_run(args.runid)
    config = config_parser.merge_configs(run.data.params)
    config = config_parser.combine_entries(config)

    # create directory for inference results
    path_results = create_model_dir(args.path_results, args.runid)

    # store validation settings
    eval_id = log_config(path_results, args.runid, config)

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
    
    # activate axonal delays for loihi compatible networks
    if config["model"]["name"] in ["LoihiRec4ptNet", "SplitLoihiRec4ptNet"]:
        config["model"]["spiking_neuron"]["delay"] = True

    # model initialization and settings
    num_bins = 2 if config["data"]["voxel"] is None else config["data"]["voxel"]

    model = SpeckNet()
    load_model = torch.load('/home/francesco/event_planar/mlruns/657431648606694428/cb600008c9444416a5bbdd90cd21e89d/artifacts/model/data/model.pth')

    model.enc1.ff1.weight.data = load_model.encoder_unet.encoders[0].recurrent_block.ff1.weight.data
    model.enc1.ff2.weight.data = load_model.encoder_unet.encoders[0].recurrent_block.ff2.weight.data
    model.enc1.rec.weight.data = load_model.encoder_unet.encoders[0].recurrent_block.rec.weight.data

    model.enc2.ff1.weight.data = load_model.encoder_unet.encoders[1].recurrent_block.ff1.weight.data
    model.enc2.ff2.weight.data = load_model.encoder_unet.encoders[1].recurrent_block.ff2.weight.data
    model.enc2.rec.weight.data = load_model.encoder_unet.encoders[1].recurrent_block.rec.weight.data

    model.pooling.weight.data = load_model.encoder_unet.pooling.conv2d.weight.data
    model.pred.weight.data = load_model.encoder_unet.pred.conv2d.weight.data

    model.to(device)

    save_state_dict(args.runid, model)
    model.eval()

    # homogrpahy projection
    homography = HomographyWarping(config, flow_scaling=config["loss"]["flow_scaling"], K=data.cam_mtx)

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
                import pdb; pdb.set_trace()
                x = model(inputs["net_input"].to(device).view(2,180,180))
                flow_vectors.append(x.reshape(-1).cpu()[0:2])

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
    test(args, YAMLParser(args.config))

    flow_vectors = torch.stack(flow_vectors, dim=1)

    flow1 = flow_vectors[0].numpy()
    flow2 = flow_vectors[1].numpy()

    window_size = 50
    smooth_flow1 = np.convolve(flow1, np.ones(window_size)/window_size, mode='valid')
    smooth_flow2 = np.convolve(flow2, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(15,15))
    plt.title(args.runid)

    plt.plot(list(range(len(flow1))), flow1)
    plt.plot(list(range(len(flow2))), flow2)


    plt.xlabel('Steps')
    plt.ylabel('Flow Components')
    plt.grid()

    directory = 'results_inference/' + args.runid
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)

    plt.savefig(directory + '/speck_flow_vectors' + '.png')
    plt.show()

