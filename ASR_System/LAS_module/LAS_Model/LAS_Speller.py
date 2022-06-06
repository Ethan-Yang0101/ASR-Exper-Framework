from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch.nn as nn
import torch


class LAS_Speller(nn.Module):

    '''create a LAS Speller (Multi-layer RNN based Decoder)'''

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, encoder_context_dim,
                 rnn_type='lstm', apply_encoder_context=False, label_smoothing_rate=0.0):
        """
        Args:
            vocab_size: number of vocabuary
            embedding_dim: embedding dimenson of vocabuary
            hidden_size: hidden size of RNN
            num_layers: number of stacked layers of RNN
            encoder_context_dim: encoder catenated hidden states dimension
            rnn_type: type of RNN model ['rnn', 'lstm', 'gru']
            apply_encoder_context: whether apply encoder context
            label_smoothing_rate: label smoothing rate
        """
        super(LAS_Speller, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_context_dim = encoder_context_dim
        self.rnn_type = rnn_type
        self.label_smoothing_rate = label_smoothing_rate
        self.apply_encoder_context = apply_encoder_context
        self.emb = nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=embedding_dim,
                                padding_idx=32)
        self.rnn = nn.ModuleList()
        if self.rnn_type == 'rnn':
            self.rnn += [nn.RNNCell(embedding_dim+hidden_size, hidden_size)]
            for _ in range(1, num_layers):
                self.rnn += [nn.RNNCell(hidden_size, hidden_size)]
        if self.rnn_type == 'gru':
            self.rnn += [nn.GRUCell(embedding_dim+hidden_size, hidden_size)]
            for _ in range(1, num_layers):
                self.rnn += [nn.GRUCell(hidden_size, hidden_size)]
        if self.rnn_type == 'lstm':
            self.rnn += [nn.LSTMCell(embedding_dim+hidden_size, hidden_size)]
            for _ in range(1, num_layers):
                self.rnn += [nn.LSTMCell(hidden_size, hidden_size)]
        context_size = hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(context_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size))
        self.encoder_map = nn.Linear(encoder_context_dim, hidden_size)

    def init_context_vectors(self, encoder_states):
        batch_size = encoder_states.size(0)
        return encoder_states.new_zeros(batch_size, self.hidden_size)

    def init_hidden_states(self, encoder_states):
        batch_size = encoder_states.size(0)
        return encoder_states.new_zeros(batch_size, self.hidden_size)

    def set_attender(self, attender):
        self.attention = attender

    def forward(self, encoder_states, encoder_context, target_sequence, target_labels):
        '''
        Args:
            encoder_states: encoder upper layer output in each time step
            [batch_size, seq_len, encoder_output_size]
            encoder_context: encoder catenated hidden states
            [batch_size, catenated_hidden_size]
            target_sequence: decoder input text sequence
            [batch_size, seq_len]
            target_labels: decoder output ground truth
            [batch_size, seq_len]
        Returns:
            loss: loss of forward propagation in a batch
            cer: character error rate of forward propagation in a batch
        notice: target sequence padded with 'eos' id and target labels padded with ignore_id=-1
        '''
        # target_sequence: [batch_size, seq_len]
        _, seq_length = target_sequence.size()
        # h_list: [layer_num, batch_size, hidden_size]
        if self.apply_encoder_context:
            h_list = [self.encoder_map(encoder_context)
                      for _ in range(self.num_layers)]
        else:
            h_list = [self.init_hidden_states(encoder_states)
                      for _ in range(self.num_layers)]
        # c_list: [layer_num, batch_size, hidden_size]
        if self.rnn_type == 'lstm':
            c_list = [self.init_hidden_states(encoder_states)
                      for _ in range(self.num_layers)]
        # context_vectors: [batch_size, hidden_size]
        context_vectors = self.init_context_vectors(encoder_states)
        # target_sequence: [seq_len, batch_size]
        target_sequence = target_sequence.permute(1, 0)
        output_vectors = []
        output_att_weights = []
        previous_prediction_score = None
        for t in range(seq_length):
            # sampling schedule (less mismatch)
            if t > 0 and torch.rand(1).item() <= 0.1:
                # batch_indices: [batch_size]
                sampler = Categorical(probs=previous_prediction_score)
                batch_indices = sampler.sample()
            else:
                # batch_indices: [batch_size]
                batch_indices = target_sequence[t]
            # batch_vectors: [batch_size, emb_dim]
            batch_vectors = self.emb(batch_indices)
            # rnn_input: [batch_size, emb_dim+hidden_size]
            rnn_input = torch.cat([batch_vectors, context_vectors], dim=1)
            if self.rnn_type == 'lstm':
                h_list[0], c_list[0] = self.rnn[0](
                    rnn_input, (h_list[0], c_list[0]))
                for l in range(1, self.num_layers):
                    h_list[l], c_list[l] = self.rnn[l](
                        h_list[l-1], (h_list[l], c_list[l]))
            else:
                h_list[0] = self.rnn[0](rnn_input, h_list[0])
                for l in range(1, self.num_layers):
                    h_list[l] = self.rnn[l](h_list[l-1], h_list[l])
            # h_t: [batch_size, hidden_size]
            h_t = h_list[-1]
            # context_vectors: [batch_size, hidden_size]
            context_vectors, att_weights = self.attention(
                encoder_state_vectors=encoder_states, query_vector=h_t)
            # prediction_vectors: [batch_size, hidden_size * 2]
            prediction_vectors = torch.cat((context_vectors, h_t), dim=1)
            # prediction_scores: [batch_size, vocab_size]
            prediction_scores = self.classifier(prediction_vectors)
            output_vectors.append(prediction_scores)
            output_att_weights.append(att_weights)
            previous_prediction_score = F.softmax(prediction_scores, dim=1)
        # output_vectors: [batch_size, seq_len, vocab_size]
        # output_att_weights: [batch_size, seq_len, frame_size]
        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)
        output_att_weights = torch.stack(
            output_att_weights).squeeze(2).permute(1, 0, 2)
        # compute loss
        loss = F.cross_entropy(output_vectors.reshape(-1, output_vectors.size(2)),
                               target_labels.view(-1), ignore_index=-1,
                               reduction='mean', label_smoothing=self.label_smoothing_rate)
        return loss, output_att_weights
