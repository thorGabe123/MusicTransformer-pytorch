from processor import encode_midi, START_IDX
import os
import pickle
from progress.bar import Bar
from collections import defaultdict
import random

def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False
    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)

def preprocess_midi_files_under(midi_folder, preprocess_folder):
    midi_paths = list(find_files_by_extensions(midi_folder, ['.mid', '.midi']))
    os.makedirs(midi_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)

    for path in Bar('Processing').iter(midi_paths):
        print(' ', end='[{}]'.format(path), flush=True)
        if os.path.exists(new_path + '.pickle'):
            continue

        try:
            data = encode_midi(path)
        except KeyboardInterrupt:
            print(' Abort')
            return
        except EOFError:
            print('EOF Error')
            return

        file_name = os.path.split(path)[1]
        new_path = os.path.join(preprocess_folder, file_name)

        with open(new_path + '.pickle', 'wb') as f:
            pickle.dump(data, f)

class Data:
    def __init__(self, dir_path):
        # Find files and set metadata labels (dictionary with song file names as keys)
        files = list(find_files_by_extensions(dir_path, ['.pickle']))
        composers = ['Bach', 'Beethoven', 'Chopin', 'Liszt', 'Mozart', 'Scarlatti', 'Schubert']
        self.metadata = {}

        for f in files:
            # Check if the filename contains any of the composer names
            no_label = True
            for idx, composer in enumerate(composers):
                if composer in f:
                    no_label = False
                    self.metadata[f] = {'composer':idx}
                    break
            if no_label:
                self.metadata[f] = {'composer': len(composers)}

    def all_data(self):
        # List to store each song's sequence and metadata
        data_with_metadata = []

        for file, metadata in self.metadata.items():
            # Load song data
            with open(file, 'rb') as f:
                data = pickle.load(f)
            
            # Add song data and metadata as a tuple to the list
            data_with_metadata.append((data, metadata))

        return data_with_metadata  # List of (song_data, metadata) tuples
    
    def train_test_data(self):
        # Step 1: Group data by composer
        composer_data = defaultdict(list)
        for item in self.all_data():
            sequence, metadata = item
            composer_idx = metadata['composer']
            composer_data[composer_idx].append(item)

        # Step 2: Split each composer's data randomly into train and test sets
        train_data = []
        test_data = []

        for composer, items in composer_data.items():
            # Shuffle items to randomize the split
            random.shuffle(items)
            
            # Determine the split point (90% for training)
            n_train = int(0.9 * len(items))
            
            # Split items for this composer into train and test sets
            train_data.extend(items[:n_train])  # First 90% for training
            test_data.extend(items[n_train:])   # Last 10% for testing

        return train_data, test_data