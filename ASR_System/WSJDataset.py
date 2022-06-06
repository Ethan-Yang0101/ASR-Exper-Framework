from torch.utils.data import Dataset
import kaldiark as kaldiark
import torch.nn as nn
import torch
import librosa
import numpy as np
import os


class WSJDataset(Dataset):

    def __init__(self, data_file_paths, data_offsets, label_file_paths, label_offsets,
                 char_dict, apply_ctc_task):
        '''
        Args:
            data_file_paths: data file paths from scp
            data_offsets: data offsets from scp
            label_file_paths: label file paths from scp
            label_offsets: label offsets from scp
            char_dict: label dictionary
            apply_ctc_task: whether apply ctc loss
        '''
        self.data_file_paths = data_file_paths
        self.data_offsets = data_offsets
        self.label_file_paths = label_file_paths
        self.label_offsets = label_offsets
        self.char_dict = char_dict
        self.apply_ctc_task = apply_ctc_task

    def __len__(self):
        return len(self.data_file_paths)

    def __getitem__(self, index):
        data_file_path = self.data_file_paths[index]
        data_offset = self.data_offsets[index]
        label_file_path = self.label_file_paths[index]
        label_offset = self.label_offsets[index]
        with open(data_file_path, 'rb') as data_reader:
            data_reader.seek(data_offset)
            mat = kaldiark.parse_feat_matrix(data_reader)
        with open(label_file_path, 'rb') as label_reader:
            label_reader.seek(label_offset)
            byte_label = label_reader.readline()
            labels = str(byte_label, 'utf-8').replace('\n', '').split(' ')
        char_dict = self.char_dict
        return mat, labels, char_dict

    @classmethod
    def make_speech_dataset(cls, char_filename, data_scp_filename, label_scp_filename,
                            apply_ctc_task):
        folder_path = '/'.join(data_scp_filename.split('/')[:-1])
        file_name = data_scp_filename.split('/')[-1]
        data_file_paths, data_offsets = read_scp_offset(folder_path, file_name)
        folder_path = '/'.join(label_scp_filename.split('/')[:-1])
        file_name = label_scp_filename.split('/')[-1]
        label_file_paths, label_offsets = read_scp_offset(
            folder_path, file_name)
        char_dict = read_char_label_of_data(char_filename)
        return cls(data_file_paths, data_offsets, label_file_paths,
                   label_offsets, char_dict, apply_ctc_task)


def read_scp_offset(folder_path, file_name):
    with open(os.path.join(folder_path, file_name), 'r') as f:
        file_paths, offsets = [], []
        for path in f:
            file_offset = path.replace('\n', '').split('/')[-1].split(':')
            file_path = file_offset[0]
            offset = int(file_offset[1])
            file_path = os.path.join(folder_path, file_path)
            file_paths.append(file_path)
            offsets.append(offset)
    return file_paths, offsets


def read_char_label_of_data(char_filename):
    char_dict = {}
    with open(char_filename, 'r') as f:
        for index, char in enumerate(f):
            char = char.replace('\n', '')
            char_dict[char] = index
    char_dict['<eos>'] = len(char_dict)
    char_dict['<sos>'] = len(char_dict)
    return char_dict


def data_preprocessing(data):
    spectrograms, input_lengths = [], []
    target_sequences, target_labels = [], []
    ground_truths, ground_lengths = [], []
    for mat, labels, char_dict in data:
        # preprocess features
        delta1 = librosa.feature.delta(mat, order=1, axis=0)
        delta2 = librosa.feature.delta(mat, order=2, axis=0)
        mat = np.concatenate((mat, delta1, delta2), axis=1)
        mat = mat.astype(np.float32)
        spectrogram = torch.from_numpy(mat)
        input_length = spectrogram.size(0)
        spectrograms.append(spectrogram)
        input_lengths.append(input_length)
        # preprocess sequences
        labels = [char_dict[label] for label in labels if label in char_dict]
        target_sequence = [char_dict['<sos>']] + labels
        target_label = labels + [char_dict['<eos>']]
        ground_truth = labels
        ground_lengths.append(len(ground_truth))
        target_sequence = torch.tensor(target_sequence)
        target_label = torch.tensor(target_label)
        ground_truth = torch.tensor(ground_truth)
        target_sequences.append(target_sequence)
        target_labels.append(target_label)
        ground_truths.append(ground_truth)
    # spectrograms: [batch_size, time, n_mels]
    spectrograms = nn.utils.rnn.pad_sequence(
        spectrograms, batch_first=True, padding_value=0.0)
    # target_sequences: [batch_size, seq_len]
    target_sequences = nn.utils.rnn.pad_sequence(
        target_sequences, batch_first=True, padding_value=char_dict['<eos>']).long()
    # target_labels: [batch_size, seq_len]
    target_labels = nn.utils.rnn.pad_sequence(
        target_labels, batch_first=True, padding_value=-1).long()
    # ground_truths: [batch_size, seq_len]
    ground_truths = nn.utils.rnn.pad_sequence(
        ground_truths, batch_first=True, padding_value=-1).long()
    return {'spectrograms': spectrograms,
            'input_lengths': input_lengths,
            'target_sequences': target_sequences,
            'target_labels': target_labels,
            'ground_truths': ground_truths,
            'ground_lengths': ground_lengths}
