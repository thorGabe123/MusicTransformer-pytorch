from pathlib import Path
import pretty_midi
import matplotlib.pyplot as plt
from math import ceil
import numpy as np

STEPS_PER_BAR = 1
SECONDS_PER_STEP = 2

def set_resolution(resolution):
    """
    Sets the resolution to be used during tokenization.

    Parameters:
    resolution (int): The resolution to be used during tokenization.

    Returns:
    None
    """
    global STEPS_PER_BAR
    global SECONDS_PER_STEP

    STEPS_PER_BAR = resolution

    BEATS_PER_MINUTE = 120
    BEATS_PER_SECOND = BEATS_PER_MINUTE / 60

    BEATS_PER_BAR = 4
    SECONDS_PER_BAR = BEATS_PER_BAR / BEATS_PER_SECOND
    TICKS_PER_BAR = SECONDS_PER_BAR

    SECONDS_PER_STEP = TICKS_PER_BAR / STEPS_PER_BAR

class MidiDataLoader:
    def __init__(self, dataset=None):
        # Set the dataset_path to the 'preprocessed' directory under the 'data' directory in the project root.
        if dataset is None:
            #Value error
            raise ValueError("Dataset path is not specified.")

        self.dataset_path = Path(dataset).resolve()
        self.midi_file_paths = self.load_midi_file_paths()        

    def from_path(path):
        """
        Creates a MidiDataLoader from a given path.

        Parameters:
        path (str): The path to the dataset.

        Returns:
        MidiDataLoader: A MidiDataLoader object.
        """
        return MidiDataLoader(dataset=path)

    def load_midi_file_paths(self):
        # List all MIDI files recursively within the dataset_path
        return list(self.dataset_path.rglob('*.mid')) + list(self.dataset_path.rglob('*.midi'))
    
    def load(self):
        """
        Loads the MIDI data from the dataset into a MidiData object.

        Returns:
        MidiData: A MidiData object containing the MIDI data.
        """
        # Return a MidiData object
        return MidiData(sorted(self.midi_file_paths))

class MidiData:
    def __init__(self, midi_file_paths):
        self.midi_files = [self.__get_midi_from_path(path) for path in midi_file_paths]

    def __get_midi_from_path(self, path):
        pm = pretty_midi.PrettyMIDI(str(path))
        return MidiFile(pm, path)
    
    def tokenize(self):
        """
        Tokenizes the MIDI data using the global resolution.

        Returns:
        list: A 2D list containing the tokens for each MIDI file.
        """
        tokens = [midifile.tokenize() for midifile in self.midi_files]
        return tokens

    
class MidiFile:
    def __init__(self, prettymidi_object, path=None):
        self.midi = prettymidi_object
        self.path = path

    def from_pretty_midi(pretty_midi_object):
        """
        Creates a MidiFile object from a pretty_midi object.

        Parameters:
        pretty_midi_object (pretty_midi.PrettyMIDI): A pretty_midi object.

        Returns:
        MidiFile: A MidiFile object
        """
        return MidiFile(pretty_midi_object)

    def from_tokens(tokens):
        """
        Creates a MidiFile object from a list of tokens.

        Parameters:
        tokens (list): A list of tokens.

        Returns:
        MidiFile: A MidiFile object
        """
        prettymidi_object = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(0)

        head = 0.
        for token in tokens:
            if token == 0:
                head += SECONDS_PER_STEP
                continue
            pitch, duration = decode_token(token)
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=head, end=head+(duration+1)*SECONDS_PER_STEP)
            instrument.notes.append(note)

        prettymidi_object.instruments.append(instrument)
        return MidiFile(prettymidi_object)
        
    def get_chords(self):
        pass

    def is_only_chords(self):
        return False
    
    def plot(self, fs=100):
        visualize(self, fs)

    def tokenize(self):
        """
        Tokenizes the MIDI file using the global resolution.

        Returns:
        list: A list containing the tokens for the MIDI file.
        """
        notes = [n for i in self.midi.instruments for n in i.notes]
        notes = sorted(notes, key=lambda n: n.start)

        head = 0.
        index = 0
        latest_start = notes[-1].end

        tokens = []

        while(index < len(notes)):
            n = notes[index]

            if n.start < head:
                index += 1
                continue

            interval_seconds = n.end - n.start
            length = interval_seconds / SECONDS_PER_STEP

            if length <= 0:
                index += 1
                continue

            if n.start == head:
                if length > STEPS_PER_BAR:
                    steps = STEPS_PER_BAR
                else:
                    steps = ceil(length)

                steps = steps - 1

                pitch = n.pitch # Integer
                tokens.append(pitch * STEPS_PER_BAR + steps + 1)
            else:
                if head > latest_start:
                    return tokens
                tokens.append(0)
                head += SECONDS_PER_STEP
                continue
            index += 1
        return tokens

def decode_token(token):
    """
    Decodes a token to extract the pitch and length.

    Parameters:
    token (int): The token to decode.

    Returns:
    tuple: A tuple containing the pitch and length.
    """
    if token <= 0:
        raise ValueError("Token must be greater than 0.")
        
    token_no_head = token - 1

    pitch = token_no_head // STEPS_PER_BAR
    length = (token_no_head % STEPS_PER_BAR)

    return pitch, length

def visualize(midi_file: MidiFile, fs=100):
    piano_roll = midi_file.midi.get_piano_roll(fs)
    
    plt.figure(figsize=(12, 4))
    plt.imshow(piano_roll, aspect='auto', interpolation='nearest', cmap='Greys')
    plt.xlabel('Time (in frames)')
    plt.ylabel('MIDI Note Number')
    plt.gca().invert_yaxis()
    plt.show()

def get_tokens():
    PATH = 'C:/Users/min/Documents/GitHub/MusicTransformer-pytorch 2024/dataset/midi/2008'

    set_resolution(8)
    loader = MidiDataLoader(PATH)
    data = loader.load()
    tokens = data.tokenize()
    tokens_flat = [i for subl in tokens for i in subl]
    return tokens_flat

def output_to_midi_notes(output_sequence, resolution=8):
    set_resolution(resolution)
    midifile = MidiFile.from_tokens(output_sequence)
    return midifile.midi.instruments[0].notes

# if __name__ == "__main__":
#     PATH = 'C:/Users/min/Documents/GitHub/MusicTransformer-pytorch 2024/dataset/Thor/niko'

#     set_resolution(8)
#     loader = MidiDataLoader(PATH)
#     data = loader.load()
#     tokens = np.array(data.tokenize())
#     tokens_flat = tokens.flatten()