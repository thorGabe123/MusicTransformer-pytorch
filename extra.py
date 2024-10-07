from model import MusicTransformer
from custom.config import config
import torch
import os
import utils
from progress.bar import Bar
import pickle
from data import Data
from custom.metrics import *
from custom.criterion import SmoothCrossEntropyLoss, CustomSchedule
import pretty_midi
from midi_processor.processor import Event, _event_seq2snote_seq, _merge_note, encode_midi, decode_midi

def load_model(model_path, config, new=True):
    model = MusicTransformer(
                embedding_dim=config.embedding_dim,
                vocab_size=config.vocab_size,
                num_layer=config.num_layers,
                max_seq=config.max_seq,
                dropout=config.dropout,
                debug=config.debug, loader_path=config.load_path
    )
    if not new:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    return model

def get_config(config, configs):
    model_dir = "config"
    config.load(model_dir, configs)
    config.pickle_dir = "dataset\\temp"
    if torch.cuda.is_available():
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')
    return config  

def preprocess_midi(midi_folder, pickle_folder):
    midi_paths = list(utils.find_files_by_extensions(midi_folder, ['.mid', '.midi']))
    os.makedirs(midi_folder, exist_ok=True)
    os.makedirs(pickle_folder, exist_ok=True)

    for path in Bar('Processing').iter(midi_paths):
        print(' ', end='[{}]'.format(path), flush=True)

        try:
            data = preprocess_midi(path)
        except KeyboardInterrupt:
            print(' Abort')
        except EOFError:
            print('EOF Error')

        with open('{}\\{}.pickle'.format('dataset\\preprocessed_midi', path.split('\\')[-1]), 'wb') as f:
            pickle.dump(data, f)

def get_event_list(model_path):
    dataset = Data(config.pickle_dir)
    metric_set = MetricsSet({
        'accuracy': CategoricalAccuracy(),
        'loss': SmoothCrossEntropyLoss(config.label_smooth, config.vocab_size, config.pad_token),
        'bucket':  LogitsBucketting(config.vocab_size)
    })
    mt = load_model(model_path, config)
    mt.to(config.device)
    mt.train()
    batch_x, batch_y = dataset.slide_seq2seq_batch(1, config.max_seq)
    batch_x = torch.from_numpy(batch_x).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
    batch_y = torch.from_numpy(batch_y).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
    sample = mt.forward(batch_x)
    metrics = metric_set(sample, batch_y)
    metrics['bucket'].shape
    output = torch.reshape(metrics['bucket'], (batch_x.shape))
    np_arr = output.tolist()
    return np_arr

def get_midi(model_path):
    dataset = Data(config.pickle_dir)
    metric_set = MetricsSet({
        'accuracy': CategoricalAccuracy(),
        'loss': SmoothCrossEntropyLoss(config.label_smooth, config.vocab_size, config.pad_token),
        'bucket':  LogitsBucketting(config.vocab_size)
    })
    mt = load_model(model_path, config)
    mt.to(config.device)
    mt.train()
    batch_x, batch_y = dataset.slide_seq2seq_batch(1, config.max_seq)
    batch_x = torch.from_numpy(batch_x).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
    batch_y = torch.from_numpy(batch_y).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
    sample = mt.forward(batch_x)
    metrics = metric_set(sample, batch_y)
    metrics['bucket'].shape
    output = torch.reshape(metrics['bucket'], (batch_x.shape))
    np_arr = output.tolist()
    
    decode_midi(np_arr, file_path="midi_output\\output.midi")
    decode_midi(batch_x.tolist(), file_path="midi_output\\input.midi")

def show_midi_events(event_int_list):
    event_sequence = [Event.from_int(idx) for idx in event_int_list]
    event_sequence
    snote_seq = _event_seq2snote_seq(event_sequence)
    note_seq = _merge_note(snote_seq)
    note_seq.sort(key=lambda x:x.start)

    mid = pretty_midi.PrettyMIDI()
    instument = pretty_midi.Instrument(1, False, "Project by Thor")
    instument.notes = note_seq

    mid.instruments.append(instument)
    for idx, _ in enumerate(event_sequence):
        print(f'{event_sequence[idx]} : {event_int_list[idx]}')