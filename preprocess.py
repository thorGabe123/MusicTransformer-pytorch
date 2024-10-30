import pickle
import os
import sys
from progress.bar import Bar
import utils
import numpy as np
from midi_processor.processor import encode_midi
from midi_processor.processor import Event, _event_seq2snote_seq, _merge_note, encode_midi, decode_midi, START_IDX
from custom.config import config


def preprocess_midi(path):
    return encode_midi(path)

def filter_note_on_events(temp_data):
    current_note_on_events = []
    current_note_off_events = []
    temp_events = []
    filtered_events = []         # The final list of events (filtered and non-filtered)
    time_shift_total = 0

    for idx in temp_data:
        # Check if the event is a 'note_on' or 'note_off' event
        if idx in np.arange(START_IDX['note_on'], START_IDX['note_off']):
            current_note_on_events.append(idx)

        elif idx in np.arange(START_IDX['note_off'], START_IDX['time_shift']):
            current_note_off_events.append(idx)

        if idx not in np.arange(START_IDX['time_shift'], START_IDX['velocity']):
            temp_events.append(idx)

        # Check if the event is a 'time_shift' event (or indicates the end of the group)
        else:
            time_shift_total += idx - START_IDX['time_shift']
            # Process current_note_on_events before a time shift event
            if current_note_on_events and time_shift_total >= 5:
                average_note_on = np.mean(current_note_on_events)  # Calculate average
                # Filter note_on events that are above the average
                filtered_notes = [note for note in temp_events if note > average_note_on - 1]
                filtered_events.extend(filtered_notes)  # Add filtered note_on events to final list
                current_note_on_events = []  # Reset for the next group of note_on events
                temp_events = []
                time_shift_total = 0

            elif current_note_on_events:
                temp_events.append(idx)

            # Add the time_shift event itself to the filtered list
            # filtered_events.append(idx)

    # Edge case: Handle remaining note_on events if no final time_shift is present
    if current_note_on_events:
        average_note_on = np.mean(current_note_on_events)
        filtered_note_on = [note for note in current_note_on_events if note > average_note_on]
        filtered_events.extend(filtered_note_on)

    return filtered_events

def preprocess_midi_files_under(midi_folder, preprocess_folder):
    config.load('config', ['config/full.yml'])
    midi_paths = list(utils.find_files_by_extensions(midi_folder, ['.mid', '.midi']))
    os.makedirs(midi_folder, exist_ok=True)
    os.makedirs(preprocess_folder, exist_ok=True)

    for path in Bar('Processing').iter(midi_paths):
        print(' ', end='[{}]'.format(path), flush=True)

        try:
            data = preprocess_midi(path)
            if len(data) <= config.max_seq:
                continue
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

if __name__ == '__main__':
    preprocess_midi_files_under(
            midi_folder=sys.argv[1],
            preprocess_folder=sys.argv[2])
