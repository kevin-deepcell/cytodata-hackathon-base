#!/usr/bin/env python3

import gzip
import torch
import numpy as np
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import argparse
import time
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from datetime import timedelta

from models import *
from vae_gru import VaeGRU
from config import run_arg_parser

def train_pl(vae, dataset):
    vae.train()
    train_size = int(0.9 * DATA_SAMPLE_NUM)
    val_size = DATA_SAMPLE_NUM - train_size
    subdataset = torch.utils.data.Subset(dataset, range(DATA_SAMPLE_NUM))
    train_set, val_set = random_split(subdataset, [train_size, val_size])
    # train_set = torch.utils.data.Subset(train_set, range(DATA_SAMPLE_NUM))
    # val_set = torch.utils.data.Subset(val_set, range(int(DATA_SAMPLE_NUM * 0.01)))

    for i in range(10):
        print(train_set[0])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=10,
        monitor="valid_loss",
        mode="min",
        # train_time_interval=timedelta(seconds=20),
    )

    trainer = pl.Trainer(
        gpus=1,
        resume_from_checkpoint=config.args.ckpt_path,
        val_check_interval=VAL_CHECK_INTERVAL,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
    )

    # TODO add num worker on linux; windows pytorch buggy due to OS error 22
    trainer.fit(vae, DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
                DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True))


if __name__ == "__main__":
    config.run_arg_parser()
    
    dataset = SmilesDataset(config.args.train_data_npy)
    language_mapping = dataset.getIndexToChar()
    
    # vae = VAE(args.hidden_size, len(language_mapping),args.embedding_dim,args.rnn_type,args.num_layers,args.bidirectional, device=DEVICE).to(DEVICE)

    # vae = VAE(decoder_type=config.DECODER_TYPE)
    if config.args.rnn_type == "lstm" or config.args.rnn_type == "transformer":
        print("using RNN type:", config.args.rnn_type)
        vae = VAE(decoder_type=config.DECODER_TYPE)
    else:
        print("using GRU VAE")
        vae = VaeGRU()

    vae.to(DEVICE)
    for name, param in vae.state_dict().items():
        print(name, param.size())

    # TRAIN THE MODEL
    # train(vae, dataset)
    train_pl(vae, dataset)

    # This will create the file that you will submit to evaluate SMILES
    # generated from a normal distribution. Note that vae.decoder must
    # be a Module
    z_1 = torch.normal(0, 1, size=(1, LATENT_DIM), device=DEVICE)
    with torch.no_grad():
        vae.decoder.eval()
        traced = torch.jit.trace(vae.decoder, z_1.to(DEVICE))
        torch.jit.save(traced, args.out)
