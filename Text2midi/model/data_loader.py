from aria.data.midi import MidiDict
# from aria.tokenizer import AbsTokenizer
# aria_tokenizer = AbsTokenizer()
import yaml
import jsonlines
import glob
import random
import os
import sys
import pickle
import json
import argparse
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
import torch
from torch.nn import functional as F
from transformers import T5Tokenizer
from spacy.lang.en import English

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class Text2MusicDataset(Dataset):
    def __init__(self, configs, captions, aria_tokenizer, mode="train", shuffle = False):
        self.mode = mode
        self.captions = captions
        if shuffle:
            random.shuffle(self.captions)

        # Path to dataset
        self.dataset_path = configs['raw_data']['raw_data_folders']['midicaps']['folder_path']

        # Artifact folder
        self.artifact_folder = configs['artifact_folder']
        # Load encoder tokenizer json file dictionary
        tokenizer_filepath = os.path.join(self.artifact_folder, "vocab.pkl")
        self.aria_tokenizer = aria_tokenizer #AbsTokenizer()
        # Load the pickled tokenizer dictionary
        with open(tokenizer_filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)

        # Load the sentencizer
        self.nlp = English()
        self.nlp.add_pipe('sentencizer')

        # Load the FLAN-T5 tokenizer and encoder
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

        # Get the maximum sequence length
        self.decoder_max_sequence_length = configs['model']['text2midi_model']['decoder_max_sequence_length']

        # Print length of dataset
        print("Length of dataset: ", len(self.captions))

    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]['caption']
        midi_filepath = os.path.join(self.dataset_path, self.captions[idx]['location'])

        # Read the MIDI file
        midi = MidiDict.from_midi(midi_filepath)
        if len(midi.note_msgs) == 0:
            aria_tokenized_midi = ["<SS>", "<E>"]        
        else:
            # Get the tokenized MIDI file
            aria_tokenized_midi = self.aria_tokenizer.tokenize(midi)
            # Add the start token
            aria_tokenized_midi = ["<SS>"] + aria_tokenized_midi

        # Drop a random number of sentences from the caption
        do_drop = random.random() > 0.5
        if do_drop:
            sentences = list(self.nlp(caption).sents)
            sent_length = len(sentences)
            if sent_length<4:
                how_many_to_drop = int(np.floor((20 + random.random()*30)/100*sent_length)) # between 20 and 50 percent of sentences
            else:
                how_many_to_drop = int(np.ceil((20 + random.random()*30)/100*sent_length)) # between 20 and 50 percent of sentences
            which_to_drop = np.random.choice(sent_length, how_many_to_drop, replace=False)
            new_sentences = [sentences[i] for i in range(sent_length) if i not in which_to_drop.tolist()]
            new_sentences = " ".join([new_sentences[i].text for i in range(len(new_sentences))]) # combine sentences back with a space
        else:
            new_sentences = caption

        # Tokenize the caption
        inputs = self.t5_tokenizer(new_sentences, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Tokenize the midi file
        tokenized_midi = [self.tokenizer[token] for token in aria_tokenized_midi if token in self.tokenizer]

        # Convert the tokenized MIDI file to a tensor and pad it to the maximum sequence length
        if len(tokenized_midi) < self.decoder_max_sequence_length:
            labels = F.pad(torch.tensor(tokenized_midi), (0, self.decoder_max_sequence_length - len(tokenized_midi))).to(torch.int64)
        else:
            labels = torch.tensor(tokenized_midi[-self.decoder_max_sequence_length:]).to(torch.int64)

        return input_ids, attention_mask, labels
        
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.normpath("../configs/config.yaml"),
                        help="Path to the config file")
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)

    caption_dataset_path = configs['raw_data']['caption_dataset_path']
    # Load the caption dataset
    with jsonlines.open(caption_dataset_path) as reader:
        captions = list(reader)

    # Load the dataset
    dataset = Text2MusicDataset(configs, captions, mode="train", shuffle = True)
    a,b,c = dataset[0]
    print(c.shape)