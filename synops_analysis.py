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

"""
synops_analysis.py

- This code is used to analyze the number of synops per layer in the model.
- An input sequence of events is entered and the total number of spikes at every times steps are calculated.   

"""

synops = []

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

    # inference loop
    end_test = False
    analyzer = SNNAnalyzer(model, dt=5.0)

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

                # model.encoder_unet.encoders[0].recurrent_block.ff.weight.data[model.encoder_unet.encoders[0].recurrent_block.ff.weight.data.abs() < 0.4] = 0    
                # model.encoder_unet.encoders[0].recurrent_block.rec.weight.data[model.encoder_unet.encoders[0].recurrent_block.rec.weight.data.abs() < 0.4] = 0
                # import pdb; pdb.set_trace()

                # forward pass
                x = model(inputs["net_input"].to(device))
                stats = analyzer.get_layer_statistics()['parameter']
                layer_names = list(stats.keys())
                synops_per_layer = []
                for name in layer_names:
                    synops_per_sec = stats[name]['synops'].item() / 0.005
                    synops_per_layer.append(synops_per_sec)

                synops.append(synops_per_layer)
            
            if end_test:
                break
    return layer_names

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
    
    os.makedirs('results_inference/' + args.runid, exist_ok=True)
    # launch testing
    layer_names = feature_analysis(args, YAMLParser(args.config))
    plt.figure(figsize=(10,10))
    plt.plot(list(range(len(synops))), synops, label=layer_names)
    plt.axhline(y=30e6)
    plt.legend()
    plt.grid()
    plt.savefig('results_inference/' + args.runid + '/synops.png')
    plt.show()
    