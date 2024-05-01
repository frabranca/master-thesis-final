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

in_spikes_0 = []
in_spikes_1 = []
in_spikes_2 = []

out_spikes_0 = []
out_spikes_1 = []
out_spikes_2 = []
features = []

def read_spikes0(module, input, output):
    input_ = input[0]
    prev_state = input[1]
    z_out, stack = output  
    in_spikes_0.append(z_out.mean())
    out_spikes_0.append(z_out.mean())

def read_spikes1(module, input, output):
    input_ = input[0]
    prev_state = input[1]
    z_out, stack = output
    in_spikes_1.append(input_.mean())
    out_spikes_1.append(z_out.mean())

def read_spikes2(module, input, output):
    input_ = input[0]
    prev_state = input[1]
    z_out, stack = output
    in_spikes_2.append(input_.mean())
    out_spikes_2.append(z_out.mean())    

def spike_activity(args, config_parser):
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
                import pdb; pdb.set_trace()
                
                read_0 = model.encoder_unet.encoders[0].recurrent_block.register_forward_hook(read_spikes0)
                read_1 = model.encoder_unet.encoders[1].recurrent_block.register_forward_hook(read_spikes1)
                read_2 = model.encoder_unet.encoders[2].recurrent_block.register_forward_hook(read_spikes2)
            
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
    spike_activity(args, YAMLParser(args.config))                            

    in_spikes_0 = torch.stack(in_spikes_0, dim=0).cpu()
    in_spikes_1 = torch.stack(in_spikes_1, dim=0).cpu()
    in_spikes_2 = torch.stack(in_spikes_2, dim=0).cpu()
    out_spikes_0 = torch.stack(out_spikes_0, dim=0).cpu()
    out_spikes_1 = torch.stack(out_spikes_1, dim=0).cpu()
    out_spikes_2 = torch.stack(out_spikes_2, dim=0).cpu()

    plt.figure(figsize=(15,15))
    plt.title(args.runid)

    plt.subplot(311)
    plt.plot(out_spikes_0, label='output layer 0', color='b')
    plt.plot(in_spikes_0, label='input layer 0', color='r')
    plt.ylabel('Mean Spike Activity')
    plt.grid()
    plt.legend(loc='best')

    plt.subplot(312)
    plt.plot(out_spikes_1, label='output layer 1', color='b')
    plt.plot(in_spikes_1, label='input layer 1', color='r')
    plt.ylabel('Mean Spike Activity')
    plt.grid()
    plt.legend(loc='best')

    plt.subplot(313)
    plt.plot(out_spikes_2, label='OUTPUT layer 2', color='b')
    plt.plot(in_spikes_2, label='INPUT layer 2', color='r')
    plt.xlabel('Steps')
    plt.ylabel('Mean Spike Activity')
    plt.grid()
    plt.legend(loc='best')

    directory = 'results_inference/' + args.runid
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)

    plt.savefig(directory + '/activity_plot' + '.png')
    plt.show()
