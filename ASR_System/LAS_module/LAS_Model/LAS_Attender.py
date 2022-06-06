import torch.nn.functional as F
import torch.nn as nn
import torch


class SingleHeadAttention(nn.Module):

    '''create a single head attention'''

    def __init__(self, encoder_state_dim, hidden_dim, share_mapping):
        '''
        Args:
            encoder_state_dim: encoder upper layer output dimension in a time step
            hidden_dim: decoder hidden state dimension
            share_mapping: whether use share weight mapping to compress encoder output
        '''
        super(SingleHeadAttention, self).__init__()
        self.encoder_state_dim = encoder_state_dim
        self.hidden_dim = hidden_dim
        self.share_mapping = share_mapping
        self.attention_type = 'single'
        self.encoder_map = nn.Linear(encoder_state_dim, hidden_dim)
        self.decoder_map = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, encoder_state_vectors, query_vector):
        '''
        Args:
            encoder_state_vectors: encoder upper layer output in each time step
            [batch_size, seq_len, encoder_output_size]
            query_vector: hidden state of current time step
            [batch_size, hidden_size]
        Returns:
            context_vectors: context vectors [batch_size, hidden_size]
            att_weights: weight of attention [batch_size, head_num, seq_len]
        '''
        # content based attention
        if self.share_mapping:
            # encoder_state_vectors: [batch_size, seq_len, hidden_size]
            encoder_state_vectors = self.encoder_map(encoder_state_vectors)
            query_vector = self.decoder_map(query_vector)
            query_vector = query_vector.unsqueeze(dim=1)
            vector_scores = self.mlp(encoder_state_vectors+query_vector).squeeze(2)
        else:
            # dot product attention
            # vector_scores: [batch_size, seq_len]
            vector_scores = torch.matmul(
                encoder_state_vectors, query_vector.unsqueeze(dim=2)).squeeze(2)
        # vector_probabilities: [batch_size, seq_len]
        vector_probabilities = F.softmax(vector_scores, dim=1)
        # att_weights: [batch_size, head_num, seq_len]
        att_weights = vector_probabilities.unsqueeze(1)
        # context_vectors: [batch_size, hidden_size]
        context_vectors = torch.matmul(encoder_state_vectors.transpose(2, 1),
                                       vector_probabilities.unsqueeze(dim=2)).squeeze(2)
        return context_vectors, att_weights
