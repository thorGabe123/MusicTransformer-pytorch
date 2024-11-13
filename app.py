import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from model import BigramLanguageModel, generated_event_length
import torch
import json
import os
from processor import ints2notes, notes2ints, notes2json, json2notes, note2midi_obj, midi_note2note, load_config
import midi_data as md

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BigramLanguageModel()
config_global = load_config(f'config/config_global.yaml')
model_version = config_global["model_version"]

if model_version == "thor":
    model.load_state_dict(torch.load('models/train_2.1258-val_2.2790.pth', map_location=device))
    piano_roll_folder_path = "C:/Users/Draco/Documents/Image-Line/FL Studio/Settings/Piano roll scripts"
elif model_version == "philip":
    model.load_state_dict(torch.load('models/train_0.0002-val_0.0002.pth', map_location=device))
    piano_roll_folder_path = "C:/Users/Draco/Documents/Image-Line/FL Studio/Settings/Piano roll scripts"
model = model.to(device)

def generate_sequence(sequence_list):
    # Set the composer metadata
    composers = ['Bach', 'Beethoven', 'Chopin', 'Liszt', 'Mozart', 'Scarlatti', 'Schubert']
    composer_name = "Beethoven"  # Choose the composer you want
    composer_index = composers.index(composer_name)

    # Prepare metadata input with the specified composer
    metadata = {
        "composer": torch.tensor([composer_index]).to(device)  # Metadata for a batch of 1
    }

    # generate from the model
    context = torch.tensor([sequence_list], device=device)
    with torch.no_grad():
        generated_sequence = model.generate(context, metadata, 200)
    return generated_sequence[0].tolist()

def save2json(int_seq, version):
    if version == "thor":
        new_notes = ints2notes(int_seq)
    if version == "philip":
        midi_notes = md.output_to_midi_notes(int_seq)
        new_notes = midi_note2note(midi_notes)
    output = notes2json(new_notes)
    with open(f"{piano_roll_folder_path}/monitored_folder/output.json", "w") as file:
        json.dump(output, file, indent=4)

class FileUpdateHandler(FileSystemEventHandler):
    def on_modified(self, event, version=model_version):
        # Only trigger if `input.json` is modified
        if not event.is_directory and os.path.basename(event.src_path) == "input.json":
            print(f"`input.json` updated: {event.src_path}")
            self.process_updated_file(event.src_path, version=version)

    def process_updated_file(self, filepath, version):
        # Load JSON data from `input.json`
        with open(filepath, "r") as file:
            try:
                data = json.load(file)  # Parse JSON
                if isinstance(data, list):  # Ensure it's a list of integers for `generate_sequence`
                    note_seq = json2notes(data)
                    if version == "thor":
                        token_seq = notes2ints(note_seq)
                    elif version == "philip":
                        midi_obj = note2midi_obj(note_seq)
                        token_seq = md.MidiFile(midi_obj).tokenize()
                    new_int_seq = generate_sequence(token_seq)
                    save2json(new_int_seq, version=version)
                    time.sleep(2)
                    os.remove(f'{piano_roll_folder_path}/monitored_folder/output.json')
                else:
                    print("Error: JSON data in `input.json` should be a list of integers.")
            except json.JSONDecodeError:
                print("Error: Failed to parse JSON in `input.json`.")

def start_monitoring(folder_path):
    event_handler = FileUpdateHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    print(f"Monitoring folder for `input.json` updates: {folder_path}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Start monitoring the folder
if __name__ == "__main__":
    start_monitoring(f'{piano_roll_folder_path}/monitored_folder')
