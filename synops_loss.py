def calculate_synops(model):
    loss = 0
    
    for encoder in model.encoder_unet.encoders:
        # ff1 = encoder.recurrent_block.ff1.weight.data.sum() / encoder.recurrent_block.threshold_ff1
        # ff2 = encoder.recurrent_block.ff2_save.mean() / encoder.recurrent_block.threshold_ff2
        # rec = encoder.recurrent_block.rec_save.mean() / encoder.recurrent_block.threshold_rec
        
        # ff1 = ff1 / sum(encoder.recurrent_block.ff1.weight.data.shape) 
        # ff2 = ff2 / sum(encoder.recurrent_block.ff2.weight.data.shape)
        # rec = rec / sum(encoder.recurrent_block.rec.weight.data.shape)

        # L1 regulatization
        ff1 = abs(encoder.recurrent_block.ff1.weight.data).sum() / sum(encoder.recurrent_block.ff1.weight.data.shape)
        ff2 = abs(encoder.recurrent_block.ff2.weight.data).sum() / sum(encoder.recurrent_block.ff2.weight.data.shape)
        rec = abs(encoder.recurrent_block.rec.weight.data).sum() / sum(encoder.recurrent_block.rec.weight.data.shape)

        loss += ff1 + ff2 + rec

    return loss