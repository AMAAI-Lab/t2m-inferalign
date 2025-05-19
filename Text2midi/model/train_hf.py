import os
from torch.cuda import is_available as cuda_available, is_bf16_supported
from torch.backends.mps import is_available as mps_available
import torch.nn as nn
import torch.optim as optim
import yaml
import json
import pickle
import os
import random
from tqdm import tqdm
from transformers import T5EncoderModel, BertModel, BertConfig, Trainer, TrainingArguments, PreTrainedModel, T5Config, T5EncoderModel, BertLMHeadModel
import torch
from torch import Tensor, argmax
from evaluate import load as load_metric
import sys
import argparse
import jsonlines
from data_loader_remi import Text2MusicDataset
from transformer_model import Transformer
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("configs/config.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f: ##args.config
    configs = yaml.safe_load(f)

batch_size = configs['training']['text2midi_model']['batch_size']
learning_rate = configs['training']['text2midi_model']['learning_rate']
epochs = configs['training']['text2midi_model']['epochs']

# Artifact folder
artifact_folder = configs['artifact_folder']
# Load remi tokenizer
tokenizer_filepath = os.path.join(artifact_folder, "vocab_remi.pkl")
# Load the tokenizer dictionary
with open(tokenizer_filepath, "rb") as f:
    tokenizer = pickle.load(f)

# Get the vocab size
vocab_size = tokenizer.vocab_size + 1
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
    decoder_input_ids = labels[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    # return input_ids, attention_mask, labels
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids,
        'labels': labels
    }

# Train test split captions
random.seed(444)
random.shuffle(captions)
train_size = int(0.8 * len(captions))
train_captions = captions[:train_size]
test_captions = captions[train_size:]

# Load the dataset
train_dataset = Text2MusicDataset(configs, train_captions, tokenizer, mode="train", shuffle = True)
print(f"Train Data length: {len(train_dataset)}")
test_dataset = Text2MusicDataset(configs, test_captions, tokenizer, mode="eval", shuffle = False)
print(f"Test Data length: {len(test_dataset)}")

# Dataloader
# train_dataset = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=5)
# test_dataset = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=5)

# Create the encoder-decoder model
class CustomEncoderDecoderModel(PreTrainedModel):
    def __init__(self, encoder, decoder, encoder_config, decoder_config):
        super().__init__(encoder_config)
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
    
    def forward(self, input_ids, decoder_input_ids, attention_mask=None, decoder_attention_mask=None, labels=None, **kwargs):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Assume the decoder can take encoder hidden states as inputs
        output = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            labels=labels
        )

        logits = output.logits

        loss = output.loss

        return {'loss': loss, 'logits': logits}

# Load the pre-trained FLAN T5 encoder and freeze its parameters
flan_t5_encoder = T5EncoderModel.from_pretrained('google/flan-t5-small')
for param in flan_t5_encoder.parameters():
    param.requires_grad = False

# Load the configurations
encoder_config = T5Config.from_pretrained('google/flan-t5-small')

# Define a configuration for the BERT decoder
config_decoder = BertConfig()
config_decoder.vocab_size = vocab_size
config_decoder.max_position_embeddings = configs['model']['text2midi_model']['decoder_max_sequence_length']
config_decoder.max_length = configs['model']['text2midi_model']['decoder_max_sequence_length']
config_decoder.bos_token_id = tokenizer["BOS_None"]
config_decoder.eos_token_id = tokenizer["EOS_None"]
config_decoder.pad_token_id = 0
config_decoder.num_hidden_layers = configs['model']['text2midi_model']['decoder_num_layers'] 
config_decoder.num_attention_heads = configs['model']['text2midi_model']['decoder_num_heads']
config_decoder.hidden_size = configs['model']['text2midi_model']['decoder_d_model']
config_decoder.intermediate_size = configs['model']['text2midi_model']['decoder_intermediate_size']

# set decoder config to causal lm
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True
config_decoder.tie_encoder_decoder = False
config_decoder.tie_word_embeddings = False

# Create a BERT model based on the configuration
custom_decoder = BertLMHeadModel(config_decoder)

# Initialize the custom model
model = CustomEncoderDecoderModel(
    encoder=flan_t5_encoder, 
    decoder=custom_decoder,
    encoder_config=encoder_config,
    decoder_config=config_decoder
)

# Print the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")

# Create config for the Trainer
USE_CUDA = cuda_available()
print(f"USE_CUDA: {USE_CUDA}")
if not cuda_available():
    FP16 = FP16_EVAL = BF16 = BF16_EVAL = False
elif is_bf16_supported():
    BF16 = BF16_EVAL = True
    FP16 = FP16_EVAL = False
else:
    BF16 = BF16_EVAL = False
    FP16 = FP16_EVAL = True
USE_MPS = not USE_CUDA and mps_available()

metrics = {metric: load_metric(metric) for metric in ["accuracy"]}

def compute_metrics(eval_pred):
    """
    Compute metrics for pretraining.

    Must use preprocess_logits function that converts logits to predictions (argmax or sampling).

    :param eval_pred: EvalPrediction containing predictions and labels
    :return: metrics
    """
    predictions, labels = eval_pred
    not_pad_mask = labels != 0
    labels, predictions = labels[not_pad_mask], predictions[not_pad_mask]
    return metrics["accuracy"].compute(predictions=predictions.flatten(), references=labels.flatten())

def preprocess_logits(logits: Tensor, _: Tensor) -> Tensor:
    """
    Preprocess the logits before accumulating them during evaluation.

    This allows to significantly reduce the memory usage and make the training tractable.
    """
    pred_ids = argmax(logits, dim=-1)  # long dtype
    return pred_ids

run_name = configs['training']['text2midi_model']['run_name']
model_dir = os.path.join(artifact_folder, run_name)
log_dir = os.path.join(model_dir, "logs")
# Clear the logs directory before training
os.system(f"rm -rf {log_dir}")

# Define the training arguments
training_args = TrainingArguments(
    output_dir=model_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_strategy="epoch",  # "steps" or "epoch"
    save_total_limit=1,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.3,
    max_grad_norm=3.0,
    weight_decay= configs['training']['text2midi_model']['weight_decay'],
    num_train_epochs=epochs,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=configs['training']['text2midi_model']['gradient_accumulation_steps'],
    # gradient_checkpointing=True,
    optim="adafactor",
    seed=444,
    logging_strategy="steps",
    logging_steps=10,
    logging_dir=log_dir,
    no_cuda=not USE_CUDA,
    fp16=FP16,
    fp16_full_eval=FP16_EVAL,
    bf16=BF16,
    bf16_full_eval=BF16_EVAL,
    load_best_model_at_end=True,
    # metric_for_best_model="loss",
    greater_is_better=False,
    report_to="tensorboard",
    run_name=run_name,
    push_to_hub=False,
    dataloader_num_workers=5
)

# # Define the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     compute_metrics=compute_metrics,
#     preprocess_logits_for_metrics=preprocess_logits,
#     # callbacks=[EarlyStoppingCallback(early_stopping_patience=30)]
# )

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=5)
    
    def get_eval_dataloader(self, eval_dataset):
        return DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=5)

    def get_test_dataloader(self, test_dataset):
        return DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=5)

# Define the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=30)]
)

# Train and save the model
train_result = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()