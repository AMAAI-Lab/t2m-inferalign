import os 
import torch.nn as nn
import torch.optim as optim
import yaml
import math
import time
from transformers import get_scheduler
import wandb
import pickle
import numpy as np
import json
import jsonlines
from tqdm import tqdm
import torch
from accelerate import DistributedDataParallelKwargs, Accelerator
from accelerate.logging import get_logger
from data_loader_remi import Text2MusicDataset
from transformer_model import Transformer
from torch.utils.data import DataLoader
import logging

logger = get_logger(__name__)

# Load config file
config_file = "../configs/config.yaml"
with open(config_file, 'r') as f:
    configs = yaml.safe_load(f)

batch_size = configs['training']['text2midi_model']['batch_size']
learning_rate = configs['training']['text2midi_model']['learning_rate']
epochs = configs['training']['text2midi_model']['epochs']
artifact_folder = configs['artifact_folder']
tokenizer_filepath = os.path.join(artifact_folder, "vocab_remi.pkl")
with open(tokenizer_filepath, "rb") as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer)
caption_dataset_path = configs['raw_data']['caption_dataset_path']

# Load the caption dataset
with jsonlines.open(caption_dataset_path) as reader:
    captions = list(reader)
    # captions = list(reader)

def collate_fn(batch):
    input_ids = [item[0].squeeze(0) for item in batch]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)    
    attention_mask = [item[1].squeeze(0) for item in batch]
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = [item[2].squeeze(0) for item in batch]
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return input_ids, attention_mask, labels

d_model = configs['model']['text2midi_model']['decoder_d_model']
nhead = configs['model']['text2midi_model']['decoder_num_heads']
num_layers = configs['model']['text2midi_model']['decoder_num_layers']
max_len = configs['model']['text2midi_model']['decoder_max_sequence_length']
use_moe = configs['model']['text2midi_model']['use_moe']
num_experts = configs['model']['text2midi_model']['num_experts']
dim_feedforward = configs['model']['text2midi_model']['decoder_intermediate_size']
gradient_accumulation_steps = configs['training']['text2midi_model']['gradient_accumulation_steps']
use_scheduler = configs['training']['text2midi_model']['use_scheduler']
checkpointing_steps = configs['training']['text2midi_model']['checkpointing_steps']
lr_scheduler_type = configs['training']['text2midi_model']['lr_scheduler_type']
num_warmup_steps = configs['training']['text2midi_model']['num_warmup_steps']
max_train_steps = configs['training']['text2midi_model']['max_train_steps']
with_tracking = configs['training']['text2midi_model']['with_tracking']
report_to = configs['training']['text2midi_model']['report_to']
output_dir = configs['training']['text2midi_model']['output_dir']
per_device_train_batch_size = configs['training']['text2midi_model']['per_device_train_batch_size']
save_every = configs['training']['text2midi_model']['save_every']

accelerator_log_kwargs = {}
if with_tracking:
    accelerator_log_kwargs["log_with"] = report_to
    # Remove the logging_dir argument in case of error
    accelerator_log_kwargs["logging_dir"] = output_dir
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision='fp16', kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)], **accelerator_log_kwargs)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_main_process:
    if output_dir is None or output_dir == "":
        output_dir = "saved/" + str(int(time.time()))
        if not os.path.exists("saved"):
            os.makedirs("saved")
        os.makedirs(output_dir, exist_ok=True)
    elif output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    os.makedirs("{}/{}".format(output_dir, "outputs"), exist_ok=True)
    accelerator.project_configuration.automatic_checkpoint_naming = False
    wandb.login()
    wandb.init(project="Text-2-Midi", settings=wandb.Settings(init_timeout=120))
accelerator.wait_for_everyone()
device = accelerator.device

with accelerator.main_process_first():
    dataset = Text2MusicDataset(configs, captions, remi_tokenizer=tokenizer, mode="train", shuffle=True)
    dataloader = DataLoader(dataset, batch_size=per_device_train_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)

model = Transformer(vocab_size, d_model, nhead, max_len, num_layers, dim_feedforward, use_moe, num_experts, device=device)
model.load_state_dict(torch.load('/root/output_test_new/epoch_68/pytorch_model.bin', map_location=device))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total number of trainable parameters: {total_params}")

optimizer = optim.Adam(model.parameters(), lr=1e-4)
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
print("num_update_steps_per_epoch", num_update_steps_per_epoch)
print("max_train_steps", max_train_steps)
if max_train_steps == 'None':
    max_train_steps = epochs * num_update_steps_per_epoch
    print("max_train_steps", max_train_steps)
    overrode_max_train_steps = True
    num_warmup_steps = 20000
elif isinstance(max_train_steps, str):
    max_train_steps = int(max_train_steps)
lr_scheduler = get_scheduler(
    name=lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=max_train_steps,
)
model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
dataloader = accelerator.prepare(dataloader)
if overrode_max_train_steps:
    max_train_steps = epochs * num_update_steps_per_epoch
epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
# checkpointing_steps = checkpointing_steps if checkpointing_steps.isdigit() else None
total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps
logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(dataset)}")
logger.info(f"  Num Epochs = {epochs}")
logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {max_train_steps}")

criterion = nn.CrossEntropyLoss()

def train_model_accelerate(model, dataloader, criterion, num_epochs, max_train_steps, optimizer=None, out_dir=None, checkpointing_steps='epoch', with_tracking=False, save_every=5, device='cpu'):
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 68
    model = model.to(device)
    model.train()
    best_loss = np.inf
    for epoch in range(starting_epoch, num_epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                encoder_input, attention_mask, tgt = batch
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
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.set_postfix({"Loss": loss.item()})
                progress_bar.update(1)
                completed_steps += 1
                if accelerator.is_main_process:
                    result = {}
                    result["epoch"] = epoch+1
                    result["step"] = completed_steps
                    result["train_loss"] = round(total_loss.item()/(gradient_accumulation_steps*completed_steps),4)
                    wandb.log(result)
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if out_dir is not None:
                        output_dir = os.path.join(out_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= max_train_steps:
                break
        if accelerator.is_main_process:    
            result = {}
            result["epoch"] = epoch+1
            result["step"] = completed_steps
            result["train_loss"] = round(total_loss.item()/len(dataloader), 4)
            result_string = "Epoch: {}, Loss Train: {}\n".format(epoch, result["train_loss"])
            accelerator.print(result_string)
            with open("{}/summary.jsonl".format(out_dir), "a") as f:
                f.write(json.dumps(result) + "\n\n")
            logger.info(result)
        if accelerator.is_main_process:
            if total_loss < best_loss:
                best_loss = total_loss
                save_checkpoint = True
            else:
                save_checkpoint = False
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and checkpointing_steps == "best":
            if save_checkpoint:
                accelerator.save_state("{}/{}".format(out_dir, "best"))
            if (epoch + 1) % save_every == 0:
                logger.info("Saving checkpoint at epoch {}".format(epoch+1))
                accelerator.save_state("{}/{}".format(out_dir, "epoch_" + str(epoch+1)))
        if accelerator.is_main_process and checkpointing_steps == "epoch":
            accelerator.save_state("{}/{}".format(out_dir, "epoch_" + str(epoch+1)))

train_model_accelerate(model, dataloader, criterion, num_epochs=epochs, max_train_steps=max_train_steps,
                       optimizer=optimizer, out_dir=output_dir, checkpointing_steps=checkpointing_steps,
                       with_tracking=with_tracking, save_every=save_every, device=device)

# torch.save(model.state_dict(), "transformer_decoder_remi_plus.pth")
# print("Model saved as transformer_decoder_remi_plus.pth")
