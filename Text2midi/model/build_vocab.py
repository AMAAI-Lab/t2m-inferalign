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


# Build the vocabulary
vocab = {}

instruments = ['piano', 'chromatic', 'organ', 'guitar', 'bass', 'strings', 'ensemble', 'brass', 'reed', 'pipe', 'synth_lead', 'synth_pad', 'synth_effect', 'ethnic', 'percussive', 'sfx', 'drum']

# Special tokens
for i in instruments:
    vocab[('prefix', 'instrument', i)] = len(vocab) + 1

# MIDI velocity range from 0 to 127
velocity = [0, 15, 30, 45, 60, 75, 90, 105, 120, 127]
# MIDI pitch range from 0 to 127
midi_pitch = list(range(0, 128))
# Onsets are quantized in 10 milliseconds up to 5 seconds
onset = list(range(0, 5001, 10))
duration = list(range(0, 5001, 10))

# Add the instrument tokens to the vocabulary
for v in velocity:
    for i in instruments:
        for p in midi_pitch:
            if i == "drum":
                continue
            else:
                vocab[(i, p, v)] = len(vocab) + 1

for p in midi_pitch:
    vocab[("drum", p)] = len(vocab) + 1

for o in onset:
    vocab[("onset", o)] = len(vocab) + 1
for d in duration:
    vocab[("dur", d)] = len(vocab) + 1

vocab["<T>"] = len(vocab) + 1
vocab["<D>"] = len(vocab) + 1
vocab["<U>"] = len(vocab) + 1
vocab["<SS>"] = len(vocab) + 1
print('vocab[<ss>]', vocab['<SS>'])
vocab["<S>"] = len(vocab) + 1
vocab["<E>"] = len(vocab) + 1
vocab["SEP"] = len(vocab) + 1

# Print the vocabulary length
print(f"Vocabulary length: {len(vocab)}")

# Save the vocabulary
vocab_path = os.path.join(artifact_folder, "vocab.pkl")
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)

print(f"Vocabulary saved to {vocab_path}")