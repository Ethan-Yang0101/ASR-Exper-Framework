import torch.nn.functional as F
import torch
import random


class BeamCoder(object):

    '''create a beam search decoder with external neuron language model'''

    def __init__(self, listener, attender, speller, language_model,
                 index_map, sos_id, eos_id, args):
        '''
        Args:
            listener: traned LAS listener
            attender: trained LAS attender
            speller: trained LAS speller
            language_model: neuron language model
            index_map: map of index to character label
            sos_id: start token index
            eos_id: end token index
            args: arguments needed
        '''
        self.listener = listener
        self.attender = attender
        self.speller = speller
        self.language_model = language_model
        self.index_map = index_map
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.args = args

    def beam_search_decoding(self, input_utterance, input_length, ground_truth=None, teach_force=False):
        '''
        Args:
            input_utterance: acoustic features to decode
            [1, seq_len, feature_dim]
            input_length: [1]
            ground_truth: input indices of decoder 
            teach_force: whether use teach forcing
        Returns:
            nbest_hyps: top best hypothesis
        '''
        encoder_outputs, _, input_length = self.listener(
            input_utterance, input_length)
        sos_id = self.sos_id  # sos_id in the vocab
        eos_id = self.eos_id  # eos_id in the vocab
        # node of search tree [1]
        node_y = encoder_outputs.new_zeros(1).long()
        # hyp: search path data structure needed for search
        hyp = {'score': 0.0, 'seq': [sos_id], 'hidden_prev': None, 'output_prev': None,
               'att_prev': None, 'logits': [], 'att_weights': []}
        # beam_size: number of beam retained in search space each time
        # best_hypo_num: number of best hypo retained in the final
        beam_size = self.args.beam_size
        best_hypo_num = self.args.best_hypo_num
        # maxlen: max number of time step to decode
        if teach_force:
            maxlen = len(ground_truth)
        else:
            maxlen = encoder_outputs.size(1)
        if self.args.decode_max_len != 0:
            maxlen = self.args.decode_max_len
        # hyps: available path in the current time step
        # ended_hyps: path with eos in the end (no more search)
        self.attender.reset()
        hyps, ended_hyps = [hyp], []
        for time_step in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                # context_vectors: [1, hidden_size]
                # att_prev: [1, frame_seq_len]
                context_vectors, att_prev = self.attender(
                    value=encoder_outputs, query=hyp['output_prev'],
                    input_lengths=input_length, att_prev=hyp['att_prev'])
                # node_y: [1, 1]
                node_y[0] = hyp['seq'][time_step]
                # embedded: [1, 1, emd_dim]
                embedded = self.speller.emb(node_y)
                embedded = embedded.unsqueeze(dim=1)
                context_vectors = context_vectors.unsqueeze(dim=1)
                # rnn_input: [1, 1, emd_dim+hidden_size]
                rnn_input = torch.cat((embedded, context_vectors), dim=2)
                # rnn_output: [1, 1, hidden_size]
                if time_step == 0:  # first time create hidden
                    rnn_output, hidden = self.speller.rnn(rnn_input)
                else:  # hidden and h_t iteratively updated
                    rnn_output, hidden = self.speller.rnn(
                        rnn_input, hyp['hidden_prev'])
                # prediction_vectors: [1, 1, hidden_size + size]
                prediction_vectors = torch.cat(
                    (rnn_output, context_vectors), dim=2)
                # prediction_scores: [1, 1, vocab_size]
                prediction_scores = self.speller.classifier(prediction_vectors)
                # prediction_prob: [1, 1, vocab_size]
                prediction_prob = F.log_softmax(prediction_scores, dim=2)
                prediction_prob = prediction_prob.squeeze(dim=1)
                # top_best_scores: [1, vocab_size]
                # top_best_ids: [1, vocab_size]
                top_best_scores, top_best_ids = torch.topk(
                    prediction_prob, beam_size, dim=1)
                # create new paths with the top best score node
                for j in range(beam_size):
                    new_hyp = {}
                    new_hyp['hidden_prev'] = (hidden[0][:], hidden[1][:])
                    new_hyp['output_prev'] = rnn_output.squeeze(dim=1)
                    new_hyp['att_prev'] = att_prev
                    if self.args.apply_noise and (time_step < maxlen - 1) and random.random() < self.args.noise_ratio:
                        j = int(torch.randint(0, prediction_prob.size(1)-1, (1,)))
                        new_hyp['seq'] = [0] * (1 + len(hyp['seq']))
                        new_hyp['seq'][:len(hyp['seq'])] = hyp['seq']
                        new_hyp['seq'][len(hyp['seq'])] = j
                        new_hyp['score'] = hyp['score'] + prediction_prob[0, j]
                    else:
                        if teach_force:
                            j = int(ground_truth[time_step])
                            new_hyp['seq'] = [0] * (1 + len(hyp['seq']))
                            new_hyp['seq'][:len(hyp['seq'])] = hyp['seq']
                            new_hyp['seq'][len(hyp['seq'])] = j
                            new_hyp['score'] = hyp['score'] + \
                                prediction_prob[0, j]
                        else:
                            new_hyp['seq'] = [0] * (1 + len(hyp['seq']))
                            new_hyp['seq'][:len(hyp['seq'])] = hyp['seq']
                            new_hyp['seq'][len(hyp['seq'])] = int(
                                top_best_ids[0, j])
                            new_hyp['score'] = hyp['score'] + \
                                top_best_scores[0, j]
                    new_hyp['att_weights'] = hyp['att_weights'][:] + \
                        [att_prev.squeeze()]
                    new_hyp['logits'] = hyp['logits'][:] + \
                        [prediction_scores.squeeze()]
                    hyps_best_kept.append(new_hyp)
                # compare all avaliable paths and keep top score paths
                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam_size]
            # check if this time step is end time step, add eos if true
            if not teach_force and time_step == maxlen - 1:
                for hyp in hyps_best_kept:
                    hyp['seq'].append(eos_id)
            # filter all eos end paths from available paths
            remained_hyps = []
            for hyp in hyps_best_kept:
                if hyp['seq'][-1] == eos_id:
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)
            if len(remained_hyps) == 0:
                break
            # end for a route of search, update current available paths
            hyps = remained_hyps
        # end of beam search, return top best hypothesis
        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[
            :min(len(ended_hyps), best_hypo_num)]
        return nbest_hyps
