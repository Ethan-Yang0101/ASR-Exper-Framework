from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence
import torch.nn as nn


class Listener(nn.Module):

    '''create a Listener (Multi-layer LSTM based Encoder)'''

    def __init__(self, input_size, hidden_size, vocab_size, dropout_rate):
        """
        Args:
            input_size: input dimension of acoustic features
            hidden_size: size of hidden state of each layer of RNN
            vocab_size: number of vocabuary
            dropout_rate: dropout rate applied to RNN
        """
        super(Listener, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.bottom_rnn1 = pLSTMLayer(input_size, hidden_size, subsample=False,
                                      dropout_rate=dropout_rate,
                                      last_layer=False)
        input_size = hidden_size * 2
        self.bottom_rnn2 = pLSTMLayer(input_size, hidden_size, subsample=True,
                                      dropout_rate=dropout_rate,
                                      last_layer=False)
        self.pRNN_layer1 = pLSTMLayer(input_size, hidden_size, subsample=True,
                                      dropout_rate=dropout_rate,
                                      last_layer=False)
        self.pRNN_layer2 = pLSTMLayer(input_size, hidden_size, subsample=False,
                                      dropout_rate=dropout_rate,
                                      last_layer=True)
        self.proj = MaskedLayer(nn.Linear(input_size, hidden_size))
        self.classifier = MaskedLayer(nn.Sequential(
            nn.Linear(hidden_size, vocab_size),
            nn.LogSoftmax(dim=1)))

    def forward(self, input_batch, input_lengths):
        rnn_output, input_lengths = self.bottom_rnn1(
            input_batch, input_lengths)
        rnn_output, input_lengths = self.bottom_rnn2(
            rnn_output, input_lengths)
        rnn_output, input_lengths = self.pRNN_layer1(
            rnn_output, input_lengths)
        rnn_output, input_lengths = self.pRNN_layer2(
            rnn_output, input_lengths)
        rnn_output, input_lengths = self.proj(
            rnn_output, input_lengths)
        log_probs, input_lengths = self.classifier(
            rnn_output, input_lengths)
        return rnn_output, log_probs, input_lengths


class pLSTMLayer(nn.Module):

    '''create a pyramidal BiLSTM layer'''

    def __init__(self, input_size, hidden_size, subsample=True,
                 dropout_rate=0.0, last_layer=False):
        super(pLSTMLayer, self).__init__()
        self.subsample = subsample
        self.last_layer = last_layer
        self.dropout = MaskedLayer(nn.Dropout(p=dropout_rate))
        self.pLSTM = nn.LSTM(input_size, hidden_size,
                             bidirectional=True,
                             batch_first=True)

    def forward(self, input_batch, input_lengths):
        input_packed = pack_padded_sequence(
            input_batch, input_lengths, batch_first=True, enforce_sorted=False)
        self.pLSTM.flatten_parameters()
        output_batch, _ = self.pLSTM(input_packed)
        output_batch, _ = pad_packed_sequence(output_batch, batch_first=True)
        if self.subsample:
            output_batch = output_batch[:, 1::2]
            input_lengths = [input_length //
                             2 for input_length in input_lengths]
        if not self.last_layer:
            output_batch, input_lengths = self.dropout(
                output_batch, input_lengths)
        return output_batch, input_lengths


class MaskedLayer(nn.Module):

    '''create a masked layer'''

    def __init__(self, layer):
        super(MaskedLayer, self).__init__()
        self.layer = layer

    def forward(self, input_batch, input_lengths):
        input_packed = pack_padded_sequence(input_batch, input_lengths, 
                                            batch_first=True,
                                            enforce_sorted=False)
        output_batch = self.layer(input_packed.data)
        output_batch = PackedSequence(output_batch, input_packed.batch_sizes,
                                      input_packed.sorted_indices,
                                      input_packed.unsorted_indices)
        output_batch, _ = pad_packed_sequence(output_batch, batch_first=True)
        return output_batch, input_lengths
