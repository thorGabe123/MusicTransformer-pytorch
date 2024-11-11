import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from model import BigramLanguageModel, generated_event_length
import torch
import json
import os
from processor import ints2notes, notes2ints, notes2json, json2notes

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BigramLanguageModel()
model.load_state_dict(torch.load('models/train_2.3394-val_2.3376.pth', map_location=device))
model = model.to(device)

def generate_sequence(sequence_list):
    # generate from the model
    context = torch.tensor([sequence_list], device=device)
    output_sequence = model.generate(context, max_new_tokens=generated_event_length)[0].tolist()
    return output_sequence

def save2json(int_seq):
    note_seq = ints2notes(int_seq)
    output = notes2json(note_seq)
    with open("monitored_folder/output.json", "w") as file:
        json.dump(output, file, indent=4)

class FileUpdateHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # Only trigger if `input.json` is modified
        if not event.is_directory and os.path.basename(event.src_path) == "input.json":
            print(f"`input.json` updated: {event.src_path}")
            self.process_updated_file(event.src_path)
        # if not event.is_directory and os.path.basename(event.src_path) == "output.json":
        #     print(f"`output.json` updated: {event.src_path}")
        #     self.delete_updated_file(event.src_path)

    def delete_updated_file(self, filepath):
        time.sleep(5)
        os.remove(filepath)

    def process_updated_file(self, filepath):
        # Load JSON data from `input.json`
        with open(filepath, "r") as file:
            try:
                data = json.load(file)  # Parse JSON
                if isinstance(data, list):  # Ensure it's a list of integers for `generate_sequence`
                    note_seq = json2notes(data)
                    int_seq = notes2ints(note_seq)
                    new_int_seq = generate_sequence(int_seq)
                    save2json(new_int_seq)
                    time.sleep(2)
                    os.remove('monitored_folder/output.json')
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
    folder_path = "monitored_folder"
    start_monitoring(folder_path)
