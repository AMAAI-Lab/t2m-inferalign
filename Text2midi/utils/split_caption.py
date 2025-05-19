import json
import os
import random
import jsonlines

def select_and_split_captions(input_path, output_dir, num_splits=6):
    with jsonlines.open(input_path) as reader:
        captions = [line for line in reader if line.get('test_set') is True]
    
    selected_captions = captions #random.sample(captions, 500)
    
    # Split the selected captions into num_splits groups
    split_size = len(selected_captions) // num_splits
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_splits - 1 else len(selected_captions)
        split_captions = selected_captions[start_idx:end_idx]
        
        output_path = os.path.join(output_dir, f'selected_captions_{i}.json')
        with open(output_path, 'w') as f:
            json.dump(split_captions, f, indent=4)
        print(f'Saved {len(split_captions)} captions to {output_path}')

if __name__ == "__main__":
    input_path = '/root/captions/train.json'
    output_dir = '/root/captions/'
    select_and_split_captions(input_path, output_dir)
