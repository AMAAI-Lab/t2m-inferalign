import os 
# print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
# import torch
# print("CUDA device count:", torch.cuda.device_count())
# print("CUDA current device:", torch.cuda.current_device())
# print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# os.environ['CUDA_VISIBLE_DEVICES']="2,3"
from torch.cuda import is_available as cuda_available, is_bf16_supported
from torch.backends.mps import is_available as mps_available
import torch.nn as nn
import torch.optim as optim
import yaml
import json
import pickle
import os
import random
import deepspeed
from tqdm import tqdm
import torch
from torch import Tensor, argmax
from evaluate import load as load_metric
import sys
import argparse
import jsonlines
from data_loader import Text2MusicDataset
from transformer_model import Transformer
from torch.utils.data import DataLoader

# Parse command line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--config", type=str, default=os.path.normpath("configs/config.yaml"),
#                     help="Path to the config file")
# parser = deepspeed.add_config_arguments(parser)
# args = parser.parse_args()
config_file = "../configs/config.yaml"
# Load config file
with open(config_file, 'r') as f: ##args.config
    configs = yaml.safe_load(f)

batch_size = configs['training']['text2midi_model']['batch_size']
learning_rate = configs['training']['text2midi_model']['learning_rate']
epochs = configs['training']['text2midi_model']['epochs']

# Artifact folder
artifact_folder = configs['artifact_folder']
# Load encoder tokenizer json file dictionary
tokenizer_filepath = os.path.join(artifact_folder, "vocab.pkl")
# Load the tokenizer dictionary
with open(tokenizer_filepath, "rb") as f:
    tokenizer = pickle.load(f)

# Get the vocab size
vocab_size = len(tokenizer)+1
print("Vocab size: ", vocab_size)

caption_dataset_path = configs['raw_data']['caption_dataset_path']
# Load the caption dataset
with jsonlines.open(caption_dataset_path) as reader:
    captions = list(reader)


def collate_fn(batch):
    """
    Collate function for the DataLoader
    :param batch: The batch
    :return: The collated batch
    """
    input_ids = [item[0].squeeze(0) for item in batch]
    # Pad or trim batch to the same length
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)    
    attention_mask = [item[1].squeeze(0) for item in batch]
    # Pad or trim batch to the same length
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = [item[2].squeeze(0) for item in batch]
    # Pad or trim batch to the same length
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return input_ids, attention_mask, labels


# Load the dataset
dataset = Text2MusicDataset(configs, captions, mode="train", shuffle = True)
data_length = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)


# Create the encoder-decoder model
# Initialize the model
d_model = configs['model']['text2midi_model']['decoder_d_model']  # Model dimension (same as FLAN-T5 encoder output dimension)
nhead = configs['model']['text2midi_model']['decoder_num_heads']     # Number of heads in the multiheadattention models
num_layers = configs['model']['text2midi_model']['decoder_num_layers']  # Number of decoder layers
max_len = configs['model']['text2midi_model']['decoder_max_sequence_length']  # Maximum length of the input sequence
use_moe = configs['model']['text2midi_model']['use_moe'] # Use mixture of experts
num_experts = configs['model']['text2midi_model']['num_experts'] # Number of experts in the mixture of experts
dim_feedforward = configs['model']['text2midi_model']['decoder_intermediate_size'] # Dimension of the feedforward network model
use_deepspeed = configs['model']['text2midi_model']['use_deepspeed'] # Use deepspeed
if use_deepspeed:
    ds_config = configs['deepspeed_config']['deepspeed_config_path']
    import deepspeed
    from deepspeed.accelerator import get_accelerator
    local_rank = int(os.environ['LOCAL_RANK']) 
    device = (torch.device(get_accelerator().device_name(), local_rank) if (local_rank > -1)
              and get_accelerator().is_available() else torch.device("cpu"))
    deepspeed.init_distributed(dist_backend='nccl')
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
else:
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print_every = 10
model = Transformer(vocab_size, d_model, nhead, max_len, num_layers, dim_feedforward, use_moe, num_experts, device=device)
# Print number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")
# Print number of trainable parameters
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_trainable_params}")
if not use_deepspeed:
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
torch.cuda.empty_cache()
def train_model(model, dataloader, criterion, num_epochs, optimizer=None, data_length=1000):   
    if use_deepspeed:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        model, optimizer, _, _ = deepspeed.initialize(model=model, 
                                                      optimizer=optimizer, 
                                                      model_parameters=model.parameters(),
                                                      config=ds_config)
    else:
        model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(total=int(data_length/batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for step, batch in enumerate(dataloader):
                if use_deepspeed:
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                
                # Get the batch
                encoder_input, attention_mask, tgt = batch
                # print(encoder_input.shape)
                encoder_input = encoder_input.to(device)
                attention_mask = attention_mask.to(device)
                tgt = tgt.to(device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                if use_moe:
                    outputs, aux_loss = model(encoder_input, attention_mask, tgt_input)
                else:
                    outputs = model(encoder_input, attention_mask, tgt_input)
                    aux_loss = 0

                loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
                loss += aux_loss
                if use_deepspeed:
                    model.backward(loss)
                    model.step()
                else:
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                if step % print_every == 0:
                    pbar.set_postfix({"Loss": loss.item()})
                    pbar.update(1)
            
            pbar.set_postfix({"Loss": total_loss / len(dataloader)})
            pbar.update(1)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")


# Train the model
if use_deepspeed:
    train_model(model, dataloader, criterion, num_epochs=epochs)
else:
    train_model(model, dataloader, criterion, num_epochs=epochs, optimizer=optimizer, data_length=data_length)

# Save the trained model
torch.save(model.state_dict(), "transformer_decoder_remi_plus.pth")
print("Model saved as transformer_decoder_remi_plus.pth")
