import yaml
import os
import argparse
import pickle
import glob
import numpy as np
import json
from tqdm import tqdm
import random
from copy import deepcopy
import sys
import pickle
from miditok import REMI, TokenizerConfig  # here we choose to use REMI
import jsonlines

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("configs/config.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

artifact_folder = configs["artifact_folder"]
raw_data_folders = configs["raw_data"]["raw_data_folders"]
caption_dataset_path = configs["raw_data"]["caption_dataset_path"]
dataset_path = configs["raw_data"]["raw_data_folders"]["lmd"]["folder_path"]

# Our parameters
BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": BEAT_RES,
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": True,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

# Creates the tokenizer
tokenizer = REMI(config)

# Load the caption dataset
with jsonlines.open(caption_dataset_path) as reader:
    captions = list(reader)

midi_paths = [os.path.join(dataset_path, captions[i]['location']) for i in range(len(captions))][0:30000]

# Builds the vocabulary with BPE
# vocab_size = 30000
# tokenizer.train(vocab_size=vocab_size, files_paths=midi_paths)

# Print the vocabulary length
print(f"Vocabulary length: {tokenizer.vocab_size}")

# Save the vocabulary
vocab_path = os.path.join(artifact_folder, "vocab_remi.pkl")
with open(vocab_path, 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"Vocabulary saved to {vocab_path}")