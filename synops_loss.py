"""
synops_loss.py

- This script is used to calculate the synops loss and use it during training.

"""

def calculate_synops(model):
    loss = 0
    
    for encoder in model.encoder_unet.encoders:
        # L1 regulatization
        ff1 = abs(encoder.recurrent_block.ff1.weight.data).sum() / sum(encoder.recurrent_block.ff1.weight.data.shape)
        ff2 = abs(encoder.recurrent_block.ff2.weight.data).sum() / sum(encoder.recurrent_block.ff2.weight.data.shape)
        rec = abs(encoder.recurrent_block.rec.weight.data).sum() / sum(encoder.recurrent_block.rec.weight.data.shape)

        loss += ff1 + ff2 + rec

    return loss