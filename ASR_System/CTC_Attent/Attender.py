import torch.nn.functional as F
import torch


class LocationAwareAttention(torch.nn.Module):

    def __init__(self, d_model, attention_type, sharpening_factor, aconv_chans=10, aconv_filts=100):
        '''
        Args:
            d_model: dimension of attention embedding
            attention_type: type of attention
            sharpening_factor: denosing factor
            aconv_chans: channel num of conv
            aconv_filts: filter num of conv
        '''
        super(LocationAwareAttention, self).__init__()
        self.mlp_enc = torch.nn.Linear(d_model, d_model, bias=True)
        self.mlp_dec = torch.nn.Linear(d_model, d_model, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, d_model, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1),
            padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(d_model, 1)
        self.sharpening_factor = sharpening_factor
        self.attention_type = attention_type
        self.d_model = d_model
        self.dunits = d_model
        self.eprojs = d_model
        self.att_dim = d_model
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, value, query, input_lengths, att_prev, last_attended_idx=None,
                backward_window=1, forward_window=3):
        '''
        Args:
            last_attended_idx: the index of the medium value of last att weights
            backward_window: lower bound of constraint
            forward_window: upper bound of constraint
        '''
        # rename parameters
        enc_hs_pad = value
        enc_hs_len = input_lengths
        dec_z = query
        # compute start
        batch = len(enc_hs_pad)
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad
            self.h_length = self.enc_h.size(1)
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)
        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)
        if att_prev is None:
            att_prev = 1.0 - make_pad_mask(enc_hs_len).to(
                device=dec_z.device, dtype=dec_z.dtype)
            att_prev = att_prev / att_prev.new(enc_hs_len).unsqueeze(-1)
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        att_conv = self.mlp_att(att_conv)
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)
        e = self.gvec(torch.tanh(
            att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)
        if self.mask is None:
            self.mask = make_pad_mask(enc_hs_len)
        e = e.masked_fill(self.mask, -float("inf"))
        if last_attended_idx is not None:
            e = _apply_attention_constraint(
                e, last_attended_idx, backward_window, forward_window)
        att_weights = F.softmax(self.sharpening_factor * e, dim=1)
        context_vectors = torch.sum(
            self.enc_h * att_weights.view(batch, self.h_length, 1), dim=1)
        return context_vectors, att_weights


def _apply_attention_constraint(e, last_attended_idx, backward_window=1, forward_window=3):
    backward_idx = last_attended_idx - backward_window
    forward_idx = last_attended_idx + forward_window
    if backward_idx > 0:
        e[:, :backward_idx] = -float("inf")
    if forward_idx < e.size(1):
        e[:, forward_idx:] = -float("inf")
    return e


def make_pad_mask(lengths):
    max_len, batch_size = max(lengths), len(lengths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    mask = mask.to(device)
    return mask
