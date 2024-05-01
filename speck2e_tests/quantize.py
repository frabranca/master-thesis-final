import mlflow
import torch
from torch.optim import *
from configs.parser import YAMLParser
from models.model import *
from utils.utils import load_model, save_state_dict
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.layers import IAFSqueeze
from make_config import make_layers
import samna
import torch
import h5py
import matplotlib.pyplot as plt
import pickle
import os
from sinabs.activation import MembraneReset
from models.submodules import ConvSpeck, ConvSpeck_typeB

def quantize_network(runid, to_upload=False):
    config_parser=YAMLParser('configs/eval_flow.yml')
    mlflow.set_tracking_uri("")
    run = mlflow.get_run(runid)
    config = config_parser.merge_configs(run.data.params)
    config = config_parser.combine_entries(config)

    # initialize settings
    device = config_parser.device
    config["loader"]["device"] = device

    # model initialization and settings
    num_bins = 2 if config["data"]["voxel"] is None else config["data"]["voxel"]
    model = eval(config["model"]["name"])(config["model"].copy(), config["data"]["crop"], num_bins)
    model = model.to(device)
    model = load_model(runid, model, device)
    save_state_dict(runid, model)
    model.eval()
    model.to('cpu')

    input_size = config['loader']['resolution'][0]
    input_shape = (2, input_size, input_size)

    samna_layers = []

    for encoder in model.encoder_unet.encoders:
        # rescale weights in encoders
        block = encoder.recurrent_block
        if isinstance(block, ConvSpeck):
            block.ff.weight.data  = (block.ff.weight.data  *  block.scale_ff).clone()
            block.rec.weight.data = (block.rec.weight.data *  block.scale_rec).clone()
        if isinstance(block, ConvSpeck_typeB):
            block.ff1.weight.data  = (block.ff1.weight.data *  block.scale_ff1).clone()
            block.ff2.weight.data  = (block.ff2.weight.data *  block.scale_ff2).clone()
            block.rec.weight.data =  (block.rec.weight.data *  block.scale_rec).clone()

        # make samna layers
        if model.encoder_unet.recurrent_block_type == 'convspeck':
            activation_ff = IAFSqueeze(batch_size=1, spike_threshold=block.threshold_ff, min_v_mem=-block.threshold_ff, reset_fn=MembraneReset())
            activation_rec = IAFSqueeze(batch_size=1, spike_threshold=block.threshold_rec, min_v_mem=-block.threshold_rec, reset_fn=MembraneReset())
            samna_layers.append(block.ff)
            samna_layers.append(activation_ff)
            samna_layers.append(block.rec)
            samna_layers.append(activation_rec)

        if model.encoder_unet.recurrent_block_type == 'convspeck_typeB':
            activation_ff1 = IAFSqueeze(batch_size=1, spike_threshold=block.threshold_ff1, min_v_mem=-block.threshold_ff1, reset_fn=MembraneReset())
            activation_ff2 = IAFSqueeze(batch_size=1, spike_threshold=block.threshold_ff2, min_v_mem=-block.threshold_ff2, reset_fn=MembraneReset())
            activation_rec = IAFSqueeze(batch_size=1, spike_threshold=block.threshold_rec, min_v_mem=-block.threshold_rec, reset_fn=MembraneReset())
        
            samna_layers.append(block.ff1)
            samna_layers.append(activation_ff1)
            samna_layers.append(block.ff2)
            samna_layers.append(activation_ff2)
            samna_layers.append(block.rec)
            samna_layers.append(activation_rec)
    
    # rescale pooling layer
    pool = model.encoder_unet.pooling
    pool.convpool.weight.data = (pool.convpool.weight.data * pool.threshold).clone()
    pool.activation = IAFSqueeze(batch_size=1, spike_threshold=pool.threshold, min_v_mem=-pool.threshold, reset_fn=MembraneReset())

    # rescale pred layer
    pred = model.encoder_unet.pred
    pred.conv2d.weight.data = (pred.conv2d.weight.data * pred.threshold).clone()
    
    # quantize network
    snn = nn.Sequential(*samna_layers, pool.convpool, pool.activation)
    dynapcnn = DynapcnnNetwork(snn, input_shape=input_shape, discretize=True)

    idx = 0
    if not to_upload:
        ''' put quantized weight on torch network and return'''
        for i, encoder in enumerate(model.encoder_unet.encoders):
            block = encoder.recurrent_block
            if isinstance(block, ConvSpeck):
                block.ff.weight.data  = dynapcnn.sequence[idx].conv_layer.weight.data
                block.rec.weight.data = dynapcnn.sequence[idx+1].conv_layer.weight.data

                thresh_quant_ff = int(dynapcnn.sequence[idx].spk_layer.spike_threshold)
                thresh_quant_rec = int(dynapcnn.sequence[idx+1].spk_layer.spike_threshold)

                block.activation_ff = IAFSqueeze(batch_size=1, spike_threshold=thresh_quant_ff, min_v_mem=-thresh_quant_ff, reset_fn=MembraneReset())
                block.activation_rec = IAFSqueeze(batch_size=1, spike_threshold=thresh_quant_rec, min_v_mem=-thresh_quant_rec, reset_fn=MembraneReset())
                idx+=2

            if isinstance(block, ConvSpeck_typeB):
                block.ff1.weight.data  = dynapcnn.sequence[idx].conv_layer.weight.data
                block.ff2.weight.data  = dynapcnn.sequence[idx+1].conv_layer.weight.data
                block.rec.weight.data = dynapcnn.sequence[idx+2].conv_layer.weight.data

                thresh_quant_ff1 = int(dynapcnn.sequence[idx].spk_layer.spike_threshold)
                thresh_quant_ff2 = int(dynapcnn.sequence[idx+1].spk_layer.spike_threshold)
                thresh_quant_rec = int(dynapcnn.sequence[idx+2].spk_layer.spike_threshold)

                block.activation_ff1 = IAFSqueeze(batch_size=1, spike_threshold=thresh_quant_ff1, min_v_mem=-thresh_quant_ff1, reset_fn=MembraneReset())
                block.activation_ff2 = IAFSqueeze(batch_size=1, spike_threshold=thresh_quant_ff2, min_v_mem=-thresh_quant_ff2, reset_fn=MembraneReset())
                block.activation_rec = IAFSqueeze(batch_size=1, spike_threshold=thresh_quant_rec, min_v_mem=-thresh_quant_rec, reset_fn=MembraneReset())
                idx+=3

        pool.convpool.weight.data = dynapcnn.sequence[6].conv_layer.weight.data 
        thresh = int(dynapcnn.sequence[6].spk_layer.spike_threshold)
        pool.activation = IAFSqueeze(batch_size=1, spike_threshold=thresh, min_v_mem=-thresh, reset_fn=MembraneReset())
    
        return model, config_parser, config

    else:  
        ''' return dynapcnn network object to upload'''
        return dynapcnn, config

if __name__ == "__main__": 
    # 2d81d5b11c55450db5e0118df629b88c
    # 14322ea85b894e52a645459acf8eaefc
    # a56eb6d230f14d71b32fe5f1d0a5e5f6
    # runid = 'f27270d55221461e804cd3de5a4ead95'
    runid = 'c01c2a2fe3444894be33cbf7d0d824c5'
    net_quantized = quantize_network(runid, to_upload=False)
