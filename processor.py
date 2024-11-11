import json
import numpy as np
from note import Note
import pretty_midi
import yaml

RANGE_NOTES = 128
RANGE_TIME_SHIFT = 64
RANGE_LEN = 64
RANGE_VEL = 100
TICKS_PER_RES = 12
TIME_PER_STEP = 1 / 16

START_IDX = {
    'notes': 0,
    'length': RANGE_NOTES,
    'time_shift': RANGE_NOTES + RANGE_LEN,
    'velocity': RANGE_NOTES + RANGE_LEN + RANGE_TIME_SHIFT
}

def load_config(filepath):
    """Loads configuration from a YAML file."""
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

def sort_by_second_element(data):
    return sorted(data, key=lambda x: x[1])

def extract_first_elements(sorted_data):
    """Extracts the first element from each sub-list in a sorted list."""
    return [int(x[0]) for x in sorted_data]

def notes2json(note_list):
    output = []
    # remove_faulty_event(note_list)
    for n in note_list:
        n.length *= TICKS_PER_RES
        n.time *= TICKS_PER_RES
        n.velocity /= RANGE_VEL
        n.length = int(n.length)
        n.time = int(n.time)
        output.append(n.to_dict())
    return output

def json2notes(json_data):
    notes = []
    for item in json_data:
        note = Note(
            value=item["value"],
            time=item["time"] // TICKS_PER_RES,
            length=item["length"] // TICKS_PER_RES,
            velocity=item["velocity"] * RANGE_VEL
        )
        notes.append(note)
    return notes

def midi_note2note(note_seq):
    value = 0
    cur_time = 0
    velocity = 0
    length = 0
    new_note_seq = []

    for note in note_seq:
        value = note.pitch
        cur_time = note.start // TIME_PER_STEP
        velocity = note.velocity // 1.28
        length = max((note.end - note.start) // TIME_PER_STEP, 1)
        n = Note(value, cur_time, velocity, length)
        new_note_seq.append(n)
    sorted_notes = sorted(new_note_seq, key=lambda note: note.time)
    return sorted_notes

def note2midi_obj(note_seq):
    prettymid_obj = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)
    new_note_seq = []

    for note in note_seq:
        # Reverse the operations from midi_note2note
        pitch = note.value
        start = note.time * TIME_PER_STEP
        velocity = int(note.velocity * 1.28)
        length = note.length
        end = start + length * TIME_PER_STEP

        # Create a new MIDI note object
        midi_note = pretty_midi.Note(pitch=pitch, start=start, end=end, velocity=velocity)
        new_note_seq.append(midi_note)

    # Sort the MIDI notes by their start time
    sorted_notes = sorted(new_note_seq, key=lambda note: note.start)
    instrument.notes = sorted_notes
    prettymid_obj.instruments.append(instrument)
    return prettymid_obj


def notes2ints(notes):
    prev_time = 0
    prev_vel = 0
    int_seq = []
    for n in notes:
        if n.time != prev_time:
            int_seq.append(min(n.time - prev_time, RANGE_TIME_SHIFT) + START_IDX['time_shift'])
            prev_time = n.time
        if n.velocity != prev_vel:
            int_seq.append(min(n.velocity, RANGE_VEL) + START_IDX['velocity'])
            prev_vel = n.velocity
        int_seq.append(min(n.value, RANGE_NOTES) + START_IDX['notes'])
        int_seq.append(min(n.length, RANGE_LEN) + START_IDX['length'])
    int_list = [int(x) for x in int_seq]
    return int_list

def ints2notes(int_seq):
    notes = []
    time = 0
    velocity = 0
    value = None
    length = None

    for i in int_seq:
        if i in range(START_IDX['time_shift'], START_IDX['time_shift'] + RANGE_TIME_SHIFT):
            # Update time by the difference indicated in the integer sequence
            time += i - START_IDX['time_shift']
        elif i in range(START_IDX['velocity'], START_IDX['velocity'] + RANGE_VEL):
            # Set the current velocity
            velocity = i - START_IDX['velocity']
        elif i in range(START_IDX['notes'], START_IDX['notes'] + RANGE_NOTES):
            # Set the note value
            value = i - START_IDX['notes']
        elif i in range(START_IDX['length'], START_IDX['length'] + RANGE_LEN):
            # Set the length of the note
            length = i - START_IDX['length']
        
        # After setting both value and length, create a Note and reset
        if value is not None and length is not None:
            notes.append(Note(velocity=velocity, value=value, time=time, length=length))
            value = None  # Reset for the next note
            length = None

    return notes

def encode_midi(path):
    notes = []
    mid = pretty_midi.PrettyMIDI(midi_file=path)

    for inst in mid.instruments:
        # channel = inst.program
        # TODO add midi program/channel info here
        notes.extend(inst.notes)

    new_notes = midi_note2note(notes)

    return notes2ints(new_notes)

def encode_fl_notes(path):
    with open(path, 'r') as file:
        data = json.load(file)
    notes = json2notes(data)

    return notes2ints(notes)

def decode_midi(int_seq, file_path=None):
    note_seq = ints2notes(int_seq)
    midi_notes = []
    for n in note_seq:
        midi_notes.append(pretty_midi.Note(velocity=int(n.velocity * 1.28), pitch=int(n.value), start=float(n.time * TIME_PER_STEP), end=float((n.time + n.length) * TIME_PER_STEP)))
    instument = pretty_midi.Instrument(1, False, "Developed By Thor Gabe")
    instument.notes = midi_notes
    mid = pretty_midi.PrettyMIDI()
    mid.instruments.append(instument)
    if file_path is not None:
        mid.write(file_path)
    return mid

def decode_fl_notes(int_seq):
    note_seq = ints2notes(int_seq)
    json_seq = notes2json(note_seq)
    with open("output.json", "w") as file:
        json.dump(json_seq, file, indent=4)
    return json_seq