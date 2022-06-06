from LAS_Model.LAS_Listener import LAS_Listener
from LAS_Model.LAS_Attender import SingleHeadAttention
from LAS_Model.LAS_Speller import LAS_Speller
import torch.nn as nn
import torch


class LAS_Model(nn.Module):

    '''create a LAS model (RNN Based Encoder Decoder Architecture with Attention)'''

    def __init__(self, listener, attender, speller, apply_ctc_task, lambda_factor):
        '''
        Args:
            listener: LAS listener model
            attender: LAS attention model
            speller: LAS speller model
            apply_ctc_task: whether use ctc loss
            lambda_factor: weight of MLT
        '''
        super(LAS_Model, self).__init__()
        self.listener = listener
        self.attender = attender
        self.speller = speller
        self.speller.set_attender(self.attender)
        self.apply_ctc_task = apply_ctc_task
        self.ctc = nn.CTCLoss(blank=33, zero_infinity=True)
        self.lambda_factor = lambda_factor

    def forward(self, input_batch, input_lengths, target_sequence, target_labels, target_lengths):
        '''
        Args:
            input_batch: a batch of padded acoustic features 
            [batch_size, seq_len, feature_dim]
            input_lengths: a batch of non-pad length numbers 
            [batch_size]
            target_sequence: decoder input text sequence
            [batch_size, seq_len]
            target_labels: decoder output ground truth
            [batch_size, seq_len]
            target_lengths: a batch of non-pad length numbers
        Returns:
            loss: loss of forward propagation in a batch
            cer: character error rate of forward propagation in a batch
        '''
        encoder_states, encoder_context, log_probs, input_lengths = self.listener(
            input_batch, input_lengths)
        att_loss, att_weights = self.speller(encoder_states, encoder_context,
                                             target_sequence, target_labels)
        if self.apply_ctc_task:
            log_probs = log_probs.permute(1, 0, 2)
            sos = target_labels.new_full((target_labels.size(0), 1), 31)
            ctc_labels = torch.cat([sos, target_labels], dim=1)
            target_lengths = [length + 1 for length in target_lengths]
            ctc_loss = self.ctc(log_probs, ctc_labels,
                                input_lengths, target_lengths)
            hybrid_loss = self.lambda_factor * ctc_loss + \
                (1 - self.lambda_factor) * att_loss
            return hybrid_loss, att_loss, ctc_loss, att_weights
        else:
            return att_loss, att_weights

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, _: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        listener = LAS_Listener(package['listener_input_size'],
                                package['listener_hidden_size'],
                                package['listener_num_layers'],
                                package['listener_vocab_size'],
                                package['listener_dropout_rate'],
                                package['listener_bidirectional'],
                                package['listener_rnn_type'])
        if package['attention_type'] == 'single':
            attender = SingleHeadAttention(package['attender_encoder_state_dim'],
                                           package['attender_hidden_dim'],
                                           package['attender_share_mapping'])
        speller = LAS_Speller(package['speller_vocab_size'],
                              package['speller_embedding_dim'],
                              package['speller_hidden_size'],
                              package['speller_num_layers'],
                              package['speller_encoder_context_dim'],
                              package['speller_rnn_type'],
                              package['speller_apply_encoder_context'],
                              package['speller_label_smoothing_rate'])
        lambda_factor = package['las_lambda_factor']
        apply_ctc_task = package['apply_ctc_task']
        # listener.flatten_parameters()
        model = cls(listener, attender, speller, apply_ctc_task, lambda_factor)
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, train_state):
        package = {
            'listener_input_size': model.listener.input_size,
            'listener_hidden_size': model.listener.hidden_size,
            'listener_num_layers': model.listener.num_layers,
            'listener_vocab_size': model.listener.vocab_size,
            'listener_dropout_rate': model.listener.dropout_rate,
            'listener_bidirectional': model.listener.bidirectional,
            'listener_rnn_type': model.listener.rnn_type,
            'attention_type': model.attender.attention_type,
            'attender_encoder_state_dim': model.attender.encoder_state_dim,
            'attender_hidden_dim': model.attender.hidden_dim,
            'attender_share_mapping': model.attender.share_mapping,
            'speller_vocab_size': model.speller.vocab_size,
            'speller_embedding_dim': model.speller.embedding_dim,
            'speller_hidden_size': model.speller.hidden_size,
            'speller_num_layers': model.speller.num_layers,
            'speller_encoder_context_dim': model.speller.encoder_context_dim,
            'speller_rnn_type': model.speller.rnn_type,
            'speller_apply_encoder_context': model.speller.apply_encoder_context,
            'speller_label_smoothing_rate': model.speller.label_smoothing_rate,
            'apply_ctc_task': model.apply_ctc_task,
            'las_lambda_factor': model.lambda_factor,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'train_state': train_state
        }
        return package
