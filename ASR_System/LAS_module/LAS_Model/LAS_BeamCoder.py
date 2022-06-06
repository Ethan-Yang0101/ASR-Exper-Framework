import torch.nn.functional as F
import torch


class LAS_BeamCoder(object):

    '''create a LAS beam search decoder with external neuron language model'''

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

    def beam_search_decoding(self, input_utterance, input_length):
        '''
        Args:
            input_utterance: acoustic features to decode 
            [1, seq_len, feature_dim]
            input_length: [1]
        Returns:
            nbest_hyps: top best hypothesis
        '''
        encoder_outputs, encoder_context, _, _ = self.listener(
            input_utterance, input_length)
        # h_list: [layer_num, 1, hidden_size]
        if self.speller.apply_encoder_context:
            h_list = [self.speller.encoder_map(encoder_context)
                      for _ in range(self.speller.num_layers)]
        else:
            h_list = [self.speller.init_hidden_states(encoder_outputs)
                      for _ in range(self.speller.num_layers)]
        # c_list: [layer_num, 1, hidden_size]
        if self.speller.rnn_type == 'lstm':
            c_list = [self.speller.init_hidden_states(encoder_outputs)
                      for _ in range(self.speller.num_layers)]
        # context_vectors: [1, hidden_size]
        context_vectors = self.speller.init_context_vectors(encoder_outputs)
        sos_id = self.sos_id  # sos_id in the vocab
        eos_id = self.eos_id  # eos_id in the vocab
        # node of search tree [1]
        node_y = encoder_outputs.new_zeros(1).long()
        # hyp: search path data structure needed for search
        if self.speller.rnn_type == 'lstm':
            hyp = {'score': 0.0, 'seq': [sos_id], 'c_prev': c_list, 'h_prev': h_list,
                   'a_prev': context_vectors, 'att_weights': []}
        else:
            hyp = {'score': 0.0, 'seq': [sos_id], 'h_prev': h_list,
                   'a_prev': context_vectors, 'att_weights': []}
        # beam_size: number of beam retained in search space each time
        # best_hypo_num: number of best hypo retained in the final
        beam_size = self.args.beam_size
        best_hypo_num = self.args.best_hypo_num
        # maxlen: max number of time step to decode
        maxlen = encoder_outputs.size(1)
        if self.args.decode_max_len != 0:
            maxlen = self.args.decode_max_len
        # hyps: available path in the current time step
        # ended_hyps: path with eos in the end (no more search)
        hyps, ended_hyps = [hyp], []
        for time_step in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                # node_y: [1, 1]
                node_y[0] = hyp['seq'][time_step]
                # embedded: [1, 1, emd_dim]
                embedded = self.speller.emb(node_y)
                # rnn_input: [1, 1, emd_dim+hidden_size]
                rnn_input = torch.cat((embedded, hyp['a_prev']), dim=1)
                if self.speller.rnn_type == 'lstm':
                    h_list[0], c_list[0] = self.speller.rnn[0](
                        rnn_input, (hyp['h_prev'][0], hyp['c_prev'][0]))
                    for l in range(1, self.speller.num_layers):
                        h_list[l], c_list[l] = self.speller.rnn[l](
                            h_list[l-1], (hyp['h_prev'][l], hyp['c_prev'][l]))
                else:
                    h_list[0] = self.speller.rnn[0](
                        rnn_input, hyp['h_prev'][0])
                    for l in range(1, self.speller.num_layers):
                        h_list[l] = self.speller.rnn[l](
                            h_list[l-1], hyp['h_prev'][l])
                # rnn_output: [1, hidden_size]
                rnn_output = h_list[-1]
                # context_vectors: [1, hidden_size]
                # att_weights: [1, head_num, frame_seq_len]
                context_vectors, att_weights = self.attender(
                    encoder_state_vectors=encoder_outputs, query_vector=rnn_output)
                # prediction_vectors: [1, hidden_size * 2]
                prediction_vectors = torch.cat(
                    (context_vectors, rnn_output), dim=1)
                # prediction_scores: [1, vocab_size]
                prediction_scores = self.speller.classifier(prediction_vectors)
                # prediction_prob: [1, vocab_size]
                prediction_prob = F.log_softmax(prediction_scores, dim=1)
                # top_best_scores: [1, vocab_size]
                # top_best_ids: [1, vocab_size]
                top_best_scores, top_best_ids = torch.topk(
                    prediction_prob, beam_size, dim=1)
                # create new paths with the top best score node
                for j in range(beam_size):
                    new_hyp = {}
                    new_hyp['h_prev'] = h_list[:]
                    if self.speller.rnn_type == 'lstm':
                        new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = context_vectors[:]
                    new_hyp['seq'] = [0] * (1 + len(hyp['seq']))
                    new_hyp['seq'][:len(hyp['seq'])] = hyp['seq']
                    new_hyp['seq'][len(hyp['seq'])] = int(top_best_ids[0, j])
                    new_hyp['score'] = hyp['score'] + top_best_scores[0, j]
                    new_hyp['att_weights'] = hyp['att_weights'][:] + \
                        [att_weights]
                    hyps_best_kept.append(new_hyp)
                # compare all avaliable paths and keep top score paths
                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam_size]
            # check if this time step is end time step, add eos if true
            if time_step == maxlen - 1:
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
