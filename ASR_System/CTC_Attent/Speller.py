from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch.nn as nn
import torch


class Speller(nn.Module):

    '''create a Speller (Multi-layer LSTM based Decoder)'''

    def __init__(self, vocab_size, embedding_dim, hidden_size, label_smoothing_rate):
        """
        Args:
            vocab_size: number of vocabuary
            embedding_dim: embedding dimenson of vocabuary
            hidden_size: hidden size of RNN
            label_smoothing_rate: label smoothing rate
        """
        super(Speller, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.label_smoothing_rate = label_smoothing_rate
        self.emb = nn.Embedding(num_embeddings=vocab_size+1,
                                embedding_dim=embedding_dim,
                                padding_idx=31)
        input_size = embedding_dim + hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        input_size = 2 * hidden_size
        self.classifier = nn.Linear(input_size, vocab_size)

    def init_context_vectors(self, encoder_states):
        batch_size, hidden_size = encoder_states.size(
            0), encoder_states.size(2)
        return encoder_states.new_zeros(batch_size, hidden_size)

    def init_hidden_states(self, encoder_states):
        batch_size, hidden_size = encoder_states.size(
            0), encoder_states.size(2)
        return encoder_states.new_zeros(batch_size, hidden_size)

    def set_attender(self, attender):
        self.attention = attender

    def forward(self, encoder_states, target_sequence, target_labels, input_lengths,
                use_sampling, sampling_rate):
        # target_sequence: [batch_size, seq_len]
        _, seq_length = target_sequence.size()
        # context_vectors: [batch_size, 1, hidden_size]
        context_vectors = None
        att_prev = None
        h_t = None
        hidden = None
        # target_sequence: [seq_len, batch_size]
        target_sequence = target_sequence.permute(1, 0)
        previous_prediction_score = None
        output_vectors = []
        output_att_weights = []
        self.attention.reset()
        for t in range(seq_length):
            # context_vectors: [batch_size, hidden_size]
            # att_weights: [batch_size, frame_size]
            context_vectors, att_prev = self.attention(
                value=encoder_states, query=h_t,
                input_lengths=input_lengths, att_prev=att_prev)
            if use_sampling and t > 0 and torch.rand(1).item() <= sampling_rate:
                # batch_indices: [batch_size]
                sampler = Categorical(probs=previous_prediction_score)
                batch_indices = sampler.sample()
            else:
                # get current time step data
                batch_indices = target_sequence[t]
            # batch_vectors: [batch_size, 1, emb_dim]
            batch_vectors = self.emb(batch_indices)
            batch_vectors = batch_vectors.unsqueeze(dim=1)
            context_vectors = context_vectors.unsqueeze(dim=1)
            # rnn_input: [batch_size, 1, emb_dim+hidden_size]
            rnn_input = torch.cat([batch_vectors, context_vectors], dim=2)
            # h_t: [batch_size, 1, hidden_size]
            if t == 0:  # first time create hidden
                h_t, hidden = self.rnn(rnn_input)
            else:  # hidden and h_t iteratively updated
                h_t, hidden = self.rnn(rnn_input, hidden)
            # prediction_vectors: [batch_size, 1, hidden_size]
            prediction_vectors = torch.cat((h_t, context_vectors), dim=2)
            # prediction_scores: [batch_size, 1, vocab_size]
            prediction_scores = self.classifier(prediction_vectors)
            # prediction_scores: [batch_size, vocab_size]
            # att_weights1: [batch_size, frame_size]
            prediction_scores = prediction_scores.squeeze(dim=1)
            previous_prediction_score = F.softmax(prediction_scores, dim=1)
            h_t = h_t.squeeze(dim=1)
            output_vectors.append(prediction_scores)
            output_att_weights.append(att_prev)
        # output_vectors: [batch_size, seq_len, vocab_size]
        # output_att_weights: [batch_size, seq_len, frame_size]
        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)
        output_att_weights = torch.stack(output_att_weights).permute(1, 0, 2)
        logits = output_vectors.reshape(-1, output_vectors.size(2))
        # compute loss
        loss = F.cross_entropy(logits, target_labels.view(-1),
                               ignore_index=-1,
                               reduction='mean',
                               label_smoothing=self.label_smoothing_rate)
        return loss, output_att_weights
