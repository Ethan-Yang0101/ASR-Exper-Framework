from CTC_Attent.Listener import Listener
from CTC_Attent.Attender import LocationAwareAttention
from CTC_Attent.Speller import Speller
import torch.nn as nn
import torch


class Model(nn.Module):

    '''create a model (LSTM Based Encoder Decoder Architecture with Attention)'''

    def __init__(self, listener, attender, speller, apply_ctc_task, lambda_factor):
        '''
        Args:
            listener: listener model
            attender: attention model
            speller: speller model
            apply_ctc_task: whether use ctc loss
            lambda_factor: weight of MLT
        '''
        super(Model, self).__init__()
        self.listener = listener
        self.attender = attender
        self.speller = speller
        self.speller.set_attender(self.attender)
        self.apply_ctc_task = apply_ctc_task
        self.ctc = nn.CTCLoss(blank=31, zero_infinity=True)
        self.lambda_factor = lambda_factor

    def forward(self, input_batch, input_lengths, target_sequence, target_labels,
                ground_truths, ground_lengths, use_sampling, sampling_rate):
        encoder_states, log_probs, input_lengths = self.listener(
            input_batch, input_lengths)
        att_loss, att_weights = self.speller(
            encoder_states, target_sequence, target_labels,
            input_lengths=input_lengths, use_sampling=use_sampling,
            sampling_rate=sampling_rate)
        log_probs = log_probs.permute(1, 0, 2)
        ctc_loss = self.ctc(log_probs, ground_truths,
                            input_lengths, ground_lengths)
        hybrid_loss = self.lambda_factor * ctc_loss + \
            (1 - self.lambda_factor) * att_loss
        return hybrid_loss, att_loss, ctc_loss, att_weights

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, _: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        listener = Listener(package['listener_input_size'],
                            package['listener_hidden_size'],
                            package['listener_vocab_size'],
                            package['listener_dropout_rate'])
        if package['attention_type'] == 'location':
            attender = LocationAwareAttention(package['attender_hidden_dim'],
                                              package['attention_type'],
                                              package['sharpening_factor'])
        speller = Speller(package['speller_vocab_size'],
                          package['speller_embedding_dim'],
                          package['speller_hidden_size'],
                          package['speller_label_smoothing_rate'])
        lambda_factor = package['las_lambda_factor']
        apply_ctc_task = package['apply_ctc_task']
        model = cls(listener, attender, speller, apply_ctc_task, lambda_factor)
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, train_state):
        package = {
            'listener_input_size': model.listener.input_size,
            'listener_hidden_size': model.listener.hidden_size,
            'listener_vocab_size': model.listener.vocab_size,
            'listener_dropout_rate': model.listener.dropout_rate,
            'attender_hidden_dim': model.attender.d_model,
            'attention_type': model.attender.attention_type,
            'sharpening_factor': model.attender.sharpening_factor,
            'speller_vocab_size': model.speller.vocab_size,
            'speller_embedding_dim': model.speller.embedding_dim,
            'speller_hidden_size': model.speller.hidden_size,
            'speller_label_smoothing_rate': model.speller.label_smoothing_rate,
            'apply_ctc_task': model.apply_ctc_task,
            'las_lambda_factor': model.lambda_factor,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'train_state': train_state
        }
        return package
