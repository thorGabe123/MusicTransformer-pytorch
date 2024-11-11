from processor import encode_midi, START_IDX
import os
import pickle
from progress.bar import Bar

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
        self.files = list(find_files_by_extensions(dir_path, ['.pickle']))

    def all_data(self):
        final_data = []
        for file in self.files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            final_data.extend(data)
            final_data.append(START_IDX['end_of_song'])
        return final_data  # batch_size, seq_len