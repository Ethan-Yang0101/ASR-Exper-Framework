from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.nn as nn
import torch


class LAS_Listener(nn.Module):

    '''create a LAS Listener (Multi-layer RNN based Encoder)'''

    def __init__(self, input_size, hidden_size, num_layers, vocab_size, dropout_rate=0.0,
                 bidirectional=True, rnn_type='lstm'):
        """
        Args:
            input_size: input dimension of acoustic features
            hidden_size: size of hidden state of each layer of RNN
            num_layers: number of stacked layers of RNN
            vocab_size: number of vocabuary
            dropout_rate: dropout rate applied to RNN
            bidirectional: whether use bidirectional RNN
            rnn_type: type of RNN model ['rnn', 'lstm', 'gru']
        """
        super(LAS_Listener, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.bottom_rnn0 = RNNLayer(input_size, hidden_size, num_layers,
                                    bidirectional=bidirectional,
                                    rnn_unit_type=rnn_type, dropout=dropout_rate)
        input_size = 2 * hidden_size if self.bidirectional else hidden_size
        self.pRNN_layer0 = pRNNLayer(input_size, hidden_size, num_layers,
                                     bidirectional=bidirectional,
                                     rnn_unit_type=rnn_type, dropout=dropout_rate)
        self.pRNN_layer1 = pRNNLayer(input_size, hidden_size, num_layers,
                                     bidirectional=bidirectional,
                                     rnn_unit_type=rnn_type, dropout=dropout_rate)
        self.pRNN_layer2 = pRNNLayer(input_size, hidden_size, num_layers,
                                     bidirectional=bidirectional,
                                     rnn_unit_type=rnn_type, dropout=dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, vocab_size))

    def forward(self, input_batch, input_lengths):
        '''
        Args:
            input_batch: a batch of padded acoustic features
            [batch_size, seq_len, feature_dim]
            input_lengths: a batch of non-pad length numbers
            [batch_size]
        Returns:
            output_batch: output of upper layer in each time step
            [batch_size, seq_len, num_directions * hidden_dim]
            out_hidden: output catenated hidden states of all layers
            [batch_size, num_directions * num_layers * hidden_dim]
        notice: input_batch must be even pad length
        '''
        # rnn_output: [batch_size, seq_len // *, num_directions * hidden_dim]
        # output_hidden: [num_directions * num_layers, batch_size, hidden_dim]
        rnn_output, _, input_lengths = self.bottom_rnn0(
            input_batch, input_lengths)
        rnn_output, _, input_lengths = self.pRNN_layer0(
            rnn_output, input_lengths)
        rnn_output, _, input_lengths = self.pRNN_layer1(
            rnn_output, input_lengths)
        rnn_output, out_hidden, input_lengths = self.pRNN_layer2(
            rnn_output, input_lengths)
        if self.rnn_type == 'lstm':
            out_hidden, _ = out_hidden
        # out_hidden: [batch_size, num_directions * num_layers, hidden_dim]
        out_hidden = out_hidden.permute(1, 0, 2)
        # out_hidden: [batch_size, num_directions * num_layers * hidden_dim]
        out_hidden = out_hidden.contiguous().view(out_hidden.size(0), -1)
        log_probs = F.log_softmax(self.classifier(rnn_output), dim=2)
        return rnn_output, out_hidden, log_probs, input_lengths


class RNNLayer(nn.Module):

    '''create a RNN based layer'''

    def __init__(self, input_size, hidden_size, num_layers, bidirectional,
                 rnn_unit_type='lstm', dropout=0.0):
        '''
        Args:
            All these parameters are overwrited rnn parts
            rnn_unit_type: choose one type of RNN
        '''
        super(RNNLayer, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit_type.upper())
        self.rnn = self.rnn_unit(input_size, hidden_size,
                                 num_layers=num_layers,
                                 bidirectional=bidirectional,
                                 dropout=dropout,
                                 batch_first=True)

    def forward(self, input_batch, input_lengths):
        '''
        Args:
            input_batch: a batch of padded acoustic features
            [batch_size, seq_len, feature_dim]
        Returns:
            output_batch: output of upper layer in each time step
            [batch_size, seq_len, num_directions * hidden_dim]
            out_hidden: output catenated hidden states of all layers
            [batch_size, num_directions * num_layers * hidden_dim]
        '''
        input_packed = pack_padded_sequence(
            input_batch, input_lengths, batch_first=True, enforce_sorted=False)
        output_batch, hidden = self.rnn(input_packed)
        output_batch, _ = pad_packed_sequence(output_batch, batch_first=True)
        return output_batch, hidden, input_lengths


class pRNNLayer(nn.Module):

    '''create a pyramid RNN based layer'''

    def __init__(self, input_size, hidden_size, num_layers, bidirectional,
                 rnn_unit_type='lstm', dropout=0.0):
        '''
        Args:
            All these parameters are overwrited rnn parts
            rnn_unit_type: choose one type of RNN
        '''
        super(pRNNLayer, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit_type.upper())
        self.pRNN = self.rnn_unit(input_size*2, hidden_size,
                                  num_layers=num_layers,
                                  bidirectional=bidirectional,
                                  dropout=dropout,
                                  batch_first=True)

    def forward(self, input_batch, input_lengths):
        '''
        Args:
            input_batch: a batch of padded acoustic features
            [batch_size, seq_len, feature_dim]
        Returns:
            output_batch: output of upper layer in each time step
            [batch_size, seq_len, num_directions * hidden_dim]
            out_hidden: output catenated hidden states of all layers
            [batch_size, num_directions * num_layers * hidden_dim]
        '''
        batch_size, seq_length, feature_dim = input_batch.size()
        # input_batch: [batch_size, seq_len / 2, feature_dim * 2]
        if seq_length % 2 != 0:
            padding = input_batch.new_zeros((batch_size, 1, feature_dim))
            input_batch = torch.cat([input_batch, padding], axis=1)
            input_lengths = [input_length+1 for input_length in input_lengths]
        input_batch = input_batch.contiguous().view(
            batch_size, input_batch.size(1) // 2, feature_dim * 2)
        input_lengths = [input_length // 2 for input_length in input_lengths]
        input_packed = pack_padded_sequence(
            input_batch, input_lengths, batch_first=True, enforce_sorted=False)
        # output_batch: [batch_size, seq_len / 2, hidden_size]
        output_batch, hidden = self.pRNN(input_packed)
        output_batch, _ = pad_packed_sequence(output_batch, batch_first=True)
        return output_batch, hidden, input_lengths
