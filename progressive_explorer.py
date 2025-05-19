import pretty_midi
import torch
import numpy as np
import os
import tempfile
import shutil
import librosa
from tqdm import tqdm
from collections import Counter
import pickle
from typing import List, Dict, Tuple, Optional, Union
import random
import time
import anthropic
from key_reward import remove_drum_track, get_beat_time, get_piano_roll, cal_key, extract_midi_notes, count_off_key_notes, all_key_names
from transformers import T5Tokenizer, ClapProcessor, ClapModel
from midi2audio import FluidSynth

# Import base classes
from Text2midi.model.transformer_model import Transformer
# from midi_explorer import MIDIExplorer

class ProgressiveExplorer:
    """Explore MIDI generation using progressive token generation with reward-guided exploration"""
    
    def __init__(self, 
                 model_path: str,
                 tokenizer_filepath: str,
                 vocab_size: int,
                 device: str = "cuda",
                 soundfont_path: str = "/usr/share/sounds/sf2/FluidR3_GM.sf2",
                 token_batch_size: int = 100,
                 num_beams: int = 5,
                 replacement_top_k: int = 2,
                 temperature: float = 0.8,
                 anthropic_api_key: str = None,
                 use_caption_mutation: bool = True,
                 num_mutations: int = 5):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.token_batch_size = token_batch_size
        self.num_beams = num_beams
        self.num_mutations = num_beams
        self.replacement_top_k = replacement_top_k
        self.temperature = temperature
        
        # Load the tokenizer
        with open(tokenizer_filepath, "rb") as f:
            self.r_tokenizer = pickle.load(f)
        
        # Initialize model
        self.model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize text tokenizer
        self.text_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        
        # Initialize reward model (CLAP)
        self.reward_model_id = "laion/larger_clap_music_and_speech"
        self.clap_processor = ClapProcessor.from_pretrained(self.reward_model_id)
        self.clap_model = ClapModel.from_pretrained(self.reward_model_id).to(self.device)
        
        # Freeze reward model parameters
        for param in self.clap_model.parameters():
            param.requires_grad = False
        
        # MIDI to audio conversion
        self.soundfont_path = soundfont_path
        if os.path.exists(self.soundfont_path):
            self.fluidsynth = FluidSynth(sound_font=self.soundfont_path)
        else:
            print(f"Warning: Soundfont not found at {self.soundfont_path}")
            self.fluidsynth = None
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="progressive_midi_")
        
        # New parameters for caption mutation
        self.use_caption_mutation = use_caption_mutation
        self.num_mutations = num_mutations
        
        # Initialize Anthropic client for caption mutation
        if anthropic_api_key:
            self.anthropic_api_key = anthropic_api_key
        else:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", " ")
        
        try:
            self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        except:
            print("Warning: Could not initialize Anthropic client. Caption mutation will be limited.")
            self.client = None
    
    def __del__(self):
        """Clean up temporary files when the object is destroyed"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    class MIDISequence:
        """Represents a MIDI sequence being progressively generated"""
        def __init__(self, caption: str, input_ids, attention_mask):
            self.caption = caption
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.tokens = []
            self.midi_path = None
            self.audio_path = None
            self.reward = None
        
        def extend_tokens(self, new_tokens):
            """Add new tokens to the sequence"""
            if isinstance(new_tokens, list):
                self.tokens.extend(new_tokens)
            else:
                self.tokens.extend(new_tokens.tolist())
        
        def get_token_count(self):
            """Get the number of tokens in the sequence"""
            return len(self.tokens)
    
    @torch.no_grad()
    def generate_tokens(self, sequence, num_tokens=100):
        """Generate additional tokens for a sequence"""
        if len(sequence.tokens) == 0:
            # Initial generation with a seed token
            tgt = None #torch.full((1, 1), 1, dtype=torch.long, device=self.device)
        else:
            # Continue generation from previous tokens
            tgt = sequence.tokens #torch.tensor([sequence.tokens], dtype=torch.long, device=self.device)
        
        # Generate tokens incrementally
        # for _ in range(num_tokens):
        output = self.model.generate(
            sequence.input_ids, 
            sequence.attention_mask,
            self.r_tokenizer,
            tgt_fin=tgt,
            # caption=caption,
            max_len=num_tokens, 
            temperature=0.9
            )
            
            # Get next token probabilities
        #     next_token_logits = output[:, -1, :]
        #     next_token_probs = torch.nn.functional.softmax(next_token_logits / self.temperature, dim=-1)
        #     next_token = torch.multinomial(next_token_probs, 1)
            
        #     # Add new token to the sequence
        #     tgt = torch.cat([tgt, next_token], dim=1)
        
        # # Return the newly generated tokens (exclude the seed token if it was just created)
        # if len(sequence.tokens) == 0:
        #     return tgt[:, 1:].cpu()
        # else:
        #     return tgt[:, len(sequence.tokens):].cpu()
        output_list = output[0].tolist()
        print(f'len of output_list: {len(output_list)}')
        # print(f' last num_tokens: {output_list[-num_tokens:]}')
        return output_list[-num_tokens:] if len(sequence.tokens) > 0 else output_list
     
    def tokens_to_midi(self, tokens, output_path, max_tokens=None):
        """Convert token sequence to MIDI file"""
        print(f'coming in to tokens_to_midi')
        try:
            if max_tokens is not None:
                if len(tokens) > max_tokens:
                    tokens = tokens[-max_tokens:]
                    print(f'changed the length of tokens to: {len(tokens)}')
                else:
                    print(f'len of tokens already: {len(tokens)}')
            midi_data = self.r_tokenizer.decode(tokens)
            midi_data.dump_midi(output_path)
            return True
        except Exception as e:
            print(f"Error converting tokens to MIDI: {e}")
            return False
    
    def compute_reward(self, sequence, original_caption=None):
        """Compute reward for a sequence using CLAP model"""
        if self.fluidsynth is None:
            print("FluidSynth not available, cannot compute reward")
            return 0.0
        
        try:
            # Create temporary files
            midi_path = os.path.join(self.temp_dir, f"seq_{id(sequence)}.mid")
            audio_path = os.path.join(self.temp_dir, f"seq_{id(sequence)}.wav")
            
            # Convert tokens to MIDI
            if not self.tokens_to_midi(sequence.tokens, midi_path):
                return 0.0
            
            # Convert MIDI to audio
            self.fluidsynth.midi_to_audio(midi_path, audio_path)
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=48000, mono=True)
            
            # Prepare inputs for CLAP model
            inputs = self.clap_processor(
                text=[original_caption] if original_caption else [sequence.caption],
                audios=[audio],
                sampling_rate=48000,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Calculate CLAP score
            with torch.no_grad():
                outputs = self.clap_model(**inputs)
                logits_per_audio = outputs.logits_per_audio
                # print(f'logits_per_audio: {logits_per_audio}')
                raw_clap_score = logits_per_audio[0].item()
                
                # Normalize the score between 0 and 1 using sigmoid
                clap_score = raw_clap_score
            
            # Save paths for future reference
            sequence.midi_path = midi_path
            sequence.audio_path = audio_path
            pm = pretty_midi.PrettyMIDI(midi_path)
            pm = remove_drum_track(pm)
            sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = get_beat_time(pm, beat_division=4)
            piano_roll = get_piano_roll(pm, sixteenth_time)
            key_name = all_key_names
            key = cal_key(piano_roll, key_name, end_ratio=0.5)
            notes = extract_midi_notes(pm)
            _, off_key_count = count_off_key_notes(notes, key[0])
            
            note_counts = Counter(notes)
            most_common_note = sorted(note_counts.most_common(1))
            note_score = (len(notes) - most_common_note[0][1])/len(notes)
            
            key_score = (len(notes) - off_key_count)/len(notes)
            return key_score*2.5 + clap_score + note_score*2.5 #+ key_score*10
            
        except Exception as e:
            print(f"Error computing reward: {e}")
            return 0.0
    
    def create_variations(self, sequences, num_variations):
        """Create variations of sequences, potentially with caption mutations"""
        variations = []
        top_captions = list(set(seq.caption for seq in sequences))
        print(f'top_captions: {top_captions}')
        
        # Include the original top sequences
        for seq in sequences:
            variations.append(seq)
        
        # If we're using caption mutation and it's time to refresh mutations (every 5 iterations)
        if self.use_caption_mutation and hasattr(self, 'iteration_count') and self.iteration_count % 1 == 0:
            print(f'Creating new caption mutations for iteration')
            # Try to generate new caption mutations from the best performing caption
            best_seq = max(sequences, key=lambda x: x.reward if x.reward is not None else -float('inf'))
            new_captions = self.mutate_caption(best_seq.caption)
            
            # Create sequences with the new captions
            for caption in new_captions:
                if len(variations) < num_variations:
                    # Tokenize the new caption
                    inputs = self.text_tokenizer(caption, return_tensors='pt').to(self.device)
                    
                    # Create a new sequence with this caption, copying tokens from the best sequence
                    new_seq = self.MIDISequence(
                        caption=caption,
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask
                    )
                    new_seq.tokens = best_seq.tokens.copy()
                    variations.append(new_seq)
        
        # Create additional variations as needed from the existing sequences
        while len(variations) < num_variations:
            source_seq = random.choice(sequences)
            new_seq = self.MIDISequence(
                source_seq.caption, 
                source_seq.input_ids, 
                source_seq.attention_mask
            )
            new_seq.tokens = source_seq.tokens.copy()
            variations.append(new_seq)
        print(f'len of variations: {len(variations)}')
        for variation in variations:
            print(f'variation caption: {variation.caption}')
        # print(f'variations: {variations}')
        
        return variations[:num_variations]
    
    def mutate_caption(self, original_caption: str) -> List[str]:
        """Generate mutations of the original caption using Claude"""
        template = f'''You are a professional music description expert. You are given an original music caption and your goal is to create {self.num_mutations} different variations that will improve the clarity and specificity of the original caption. The mutations should retain the same musical style, instruments, and mood as the original.

        Hint: Think of adding more details about tempo, key signature, instruments, or emotional qualities. You can also reformulate the phrasing to be more precise.

        Only give the mutated captions in a numbered list.

        Original caption: A calm piano piece with strings
        1. A serene piano composition accompanied by gentle string arrangements in G major
        2. A peaceful piano melody with flowing string sections creating a tranquil atmosphere
        3. A soothing piano ballad enhanced by warm string harmonies at 70 BPM
        4. A relaxing piano-led instrumental with delicate string embellishments in a minor key
        5. A meditative piano piece with supporting string orchestra creating a calm ambiance

        Original caption: {original_caption}
        '''
        
        try:
            if self.client:
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=200,
                    temperature=0.8,
                    messages=[
                        {"role": "user", "content": template}
                    ]
                )
                content = response.content[0].text
                
                # Extract mutations from response
                mutations = [line.strip() for line in content.split("\n") if line.strip()]
                
                # Filter out numbering and any explanatory text
                mutations = [m.split(". ", 1)[-1].strip('"\'') for m in mutations if ". " in m]
                
                # Make sure we have the right number of mutations
                mutations = mutations[:self.num_mutations]
                
                # Add the original caption as one of the mutations if not already included
                if original_caption not in mutations:
                    mutations = [original_caption] + mutations[:-1]
                    
                return mutations
            else:
                raise Exception("Anthropic client not available")
                
        except Exception as e:
            print(f"Error in caption mutation: {e}")
            # Fallback: return original with minor variations
            return [original_caption] + [f"{original_caption} {suffix}" for suffix in 
                    ["with emotional depth", "with rhythmic elements", "with melodic focus", 
                     "with ambient qualities"]][:self.num_mutations-1]
    
    def progressive_generate(self, caption, max_tokens=2000):
        """Generate MIDI progressively with reward-based exploration"""
        print(f"Starting progressive generation for: {caption}")
        orig_caption = caption
        
        # Generate caption mutations if enabled
        if self.use_caption_mutation:
            captions = self.mutate_caption(caption)
            print("Generated caption mutations:")
            for i, mutated_caption in enumerate(captions):
                print(f"{i+1}. {mutated_caption}")
        else:
            captions = [caption]
        
        sequences = []
        
        # Create initial sequences with variations of the caption
        for caption_variant in captions:
            # Tokenize the caption variant
            inputs = self.text_tokenizer(caption_variant, return_tensors='pt').to(self.device)
            
            # Create a sequence for this caption
            sequence = self.MIDISequence(
                caption=caption_variant,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            sequences.append(sequence)
        
        # If we need more sequences than captions, create duplicates
        while len(sequences) < self.num_beams:
            # Select a random sequence to duplicate
            source_seq = random.choice(sequences)
            
            # Create a duplicate (will result in different generation later)
            new_seq = self.MIDISequence(
                caption=source_seq.caption,
                input_ids=source_seq.input_ids,
                attention_mask=source_seq.attention_mask
            )
            sequences.append(new_seq)
        
        # If we have too many sequences, trim the list
        if len(sequences) > self.num_beams:
            sequences = sequences[:self.num_beams]
        
        # Progressive generation loop
        iterations = max_tokens // self.token_batch_size + 2
        
        for iteration in tqdm(range(iterations), desc="Generating music"):
            print(f"\nIteration {iteration+1}/{iterations}, generating tokens {iteration*self.token_batch_size+1}-{(iteration+1)*self.token_batch_size}")
            
            # Generate next token batch for each sequence
            for seq in sequences:
                new_tokens = self.generate_tokens(seq, self.token_batch_size)
                seq.extend_tokens(new_tokens)
            
            # Compute rewards for all sequences
            for i, seq in enumerate(sequences):
                reward = self.compute_reward(seq, orig_caption)
                seq.reward = reward
                print(f"Sequence {i+1}: Reward = {reward:.4f}")
            
            # Skip replacement on the final iteration
            if iteration < iterations - 1:
                # Select top-k sequences
                sequences.sort(key=lambda seq: seq.reward if seq.reward is not None else -float('inf'), reverse=True)
                top_sequences = sequences[:self.replacement_top_k]
                
                # Create variations for next iteration
                sequences = self.create_variations(top_sequences, self.num_beams)
                
            # Show current best sequence
            best_seq = max(sequences, key=lambda seq: seq.reward if seq.reward is not None else -float('inf'))
            print(f"Current best reward: {best_seq.reward:.4f}")
        
        # Select best sequence after all iterations
        best_sequence = max(sequences, key=lambda seq: seq.reward if seq.reward is not None else -float('inf'))
        # print(f'lenth of best_sequence.tokens before: {len(best_sequence.tokens)}')
        best_sequence.tokens = best_sequence.tokens[-max_tokens:]
        # print(f'length of best_sequence.tokens after: {len(best_sequence.tokens)}')
        
        # Save final output
        final_midi_path = f"outputs/progressive_midi_{int(time.time())}.mid"
        os.makedirs(os.path.dirname(final_midi_path), exist_ok=True)
        
        if best_sequence.midi_path and os.path.exists(best_sequence.midi_path):
            print(f'Copying existing MIDI file to {final_midi_path}')
            self.tokens_to_midi(best_sequence.tokens, best_sequence.midi_path, max_tokens=max_tokens)
            shutil.copy(best_sequence.midi_path, final_midi_path)
        else:
            print(f'generating midi from tokens')
            self.tokens_to_midi(best_sequence.tokens, final_midi_path, max_tokens=max_tokens)
        
        print(f"\nGeneration complete!")
        print(f"Best reward: {best_sequence.reward:.4f}")
        print(f"Output MIDI saved to: {final_midi_path}")
        
        return {
            "midi_path": final_midi_path,
            "reward": best_sequence.reward,
            "tokens": best_sequence.tokens
        }

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Progressive MIDI Explorer - Generate optimized MIDI incrementally")
    parser.add_argument("--caption", type=str, required=True, help="Caption to generate MIDI from")
    parser.add_argument("--model_path", type=str, default="/root/output_test_new/epoch_30/pytorch_model.bin", 
                        help="Path to the trained transformer model")
    parser.add_argument("--tokenizer_path", type=str, default="./Text2midi/artifacts/vocab_remi.pkl",
                        help="Path to the tokenizer")
    parser.add_argument("--max_tokens", type=int, default=2000, help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=500, help="Number of tokens to generate before checking rewards")
    parser.add_argument("--beams", type=int, default=3, help="Number of parallel sequences to generate")
    parser.add_argument("--output", type=str, default="outputs/me_example_explore.mid", help="Output MIDI file path")
    
    args = parser.parse_args()
    
    # Load the tokenizer to get vocab size
    with open(args.tokenizer_path, "rb") as f:
        r_tokenizer = pickle.load(f)
    vocab_size = len(r_tokenizer)
    
    # Create explorer
    explorer = ProgressiveExplorer(
        model_path=args.model_path,
        tokenizer_filepath=args.tokenizer_path,
        vocab_size=vocab_size,
        token_batch_size=args.batch_size,
        num_beams=args.beams
    )
    
    # Generate MIDI
    result = explorer.progressive_generate(
        caption=args.caption,
        max_tokens=args.max_tokens
    )
    
    # Copy output to specified path if different
    if result["midi_path"] != args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        shutil.copy(result["midi_path"], args.output)
        print(f"Copied MIDI to: {args.output}")
