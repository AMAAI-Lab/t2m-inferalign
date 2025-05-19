import os
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

soundfont_filepath = "/root/soundfont/soundfont.sf"

def save_wav(midi_filepath, wav_filepath):
    # Check if the .wav file already exists
    if os.path.isfile(wav_filepath):
        print(f"{wav_filepath} already exists, skipping")
        return wav_filepath
    else:
        print(f"Creating {wav_filepath} from {midi_filepath}")
        
        # Run the fluidsynth command to convert MIDI to WAV
        command = f"fluidsynth -r 48000 {soundfont_filepath} -g 1.0 --quiet --no-shell {midi_filepath} -T wav -F {wav_filepath}"
        print(f"Running command: {command}")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error converting {midi_filepath} to {wav_filepath}: {stderr.decode('utf-8')}")
        else:
            print(f"Successfully created {wav_filepath}")

        return wav_filepath

def process_midi_file(midi_filepath):
    # Determine the corresponding wav file path
    relative_path = os.path.relpath(midi_filepath, "/root/Text2midi/res_acc")
    wav_filepath = os.path.join("/root/wav", relative_path.replace('.mid', '.wav'))
    wav_directory = os.path.dirname(wav_filepath)
    
    # Ensure the directory exists
    os.makedirs(wav_directory, exist_ok=True)
    
    # Convert the MIDI file to WAV
    save_wav(midi_filepath, wav_filepath)

def main():
    # Find all .mid files in /root/Text2midi/res_acc
    midi_files = []
    for root, _, files in os.walk("/root/Text2midi/res_acc"):
        for file in files:
            if file.endswith(".mid"):
                midi_files.append(os.path.join(root, file))
    
    # Use half of the available CPU cores for multiprocessing
    num_cores = cpu_count() // 2
    with Pool(num_cores) as pool:
        list(tqdm(pool.imap(process_midi_file, midi_files), total=len(midi_files), desc="Processing MIDI files"))

if __name__ == "__main__":
    main()
