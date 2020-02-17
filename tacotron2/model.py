#: *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import math
import os
import torch
from torch import nn
from torch.nn import functional as F
import sys
from os.path import abspath, dirname
# enabling modules discovery from global entrypoint
sys.path.append(abspath(dirname(__file__)+'/../'))
from common.layers import ConvNorm, LinearNorm
from common.utils import to_gpu, get_mask_from_lengths


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        self.location_conv = ConvNorm(1, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=int((attention_kernel_size - 1) / 2),
                                      stride=1, dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cum):
        processed_attention_weights = self.location_conv(attention_weights_cum)
        processed_attention_weights = processed_attention_weights.transpose(1, 2)
        processed_attention_weights = self.location_dense(processed_attention_weights)
        return processed_attention_weights


class Attention(nn.Module):
    def __init__(self, query_dim, memory_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(query_dim, attention_dim, w_init_gain='tanh')
        self.memory_layer = LinearNorm(memory_dim, attention_dim, w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, memory, attention_weights_cum):
        """
        PARAMS
        ------
        query: decoder output (B, decoder_dim)
        memory: encoder outputs (B, T_in, embed_dim)
        attention_weights_cum: cumulative attention weights (B, 1, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        # [B, T_in, attn_dim]
        key = self.memory_layer(memory)
        # [B, 1, attn_dim]
        query = self.query_layer(query.unsqueeze(1))
        # [B, T, attn_dim]
        location_sensitive_weights = self.location_layer(attention_weights_cum)
        # score function
        energies = self.v(torch.tanh(query + location_sensitive_weights + key))
        energies = energies.squeeze(-1)

        return energies

    def forward(self, query, memory, attention_weights_cum, mask=None):
        """
        PARAMS
        ------
        query: attention rnn last output [B, decoder_dim]
        memory: encoder outputs [B, T_in, embed_dim]
        attention_weights_cum: cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(query, memory, attention_weights_cum)

        if mask is not None:
            alignment.masked_fill_(mask, self.score_mask_value)

        # [B, T_in]
        attention_weights = F.softmax(alignment, dim=1)
        # [B, 1, T_in] * [B, T_in, embbed_dim]
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        # [B, embbed_dim]
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size) for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x, inference=False):
        if inference:
            for linear in self.layers:
                x = F.relu(linear(x), inplace=True)
                x0 = x[0].unsqueeze(0)
                mask = torch.bernoulli(x0.new(x0.size()).fill_(0.5))
                mask = mask.expand(x.size())
                x = x*mask*2
        else:
            for linear in self.layers:
                x = F.dropout(F.relu(linear(x), inplace=True), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, n_mel_channels, postnet_embedding_dim,
                 postnet_kernel_size, postnet_n_convolutions):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels, postnet_embedding_dim,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(postnet_embedding_dim))
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size, stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim, n_mel_channels,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(n_mel_channels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))
        return self.convolutions[-1](x)


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, encoder_n_convolutions, encoder_embedding_dim, encoder_kernel_size):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(encoder_embedding_dim,
                         encoder_embedding_dim,
                         kernel_size=encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.encoder_lstm = nn.LSTM(encoder_embedding_dim,
                                    int(encoder_embedding_dim / 2), 1,
                                    batch_first=True, bidirectional=True)

    def forward(self, x, text_lengths):
        for conv in self.convolutions:
            x = F.relu(conv(x), inplace=True)

        # [B, encoder_dim, T_in] -> [B, T_in, encoder_dim]
        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        text_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, text_lengths, batch_first=True)
        # [B, T_in, encoder_dim]
        outputs, _ = self.encoder_lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs


class Decoder(nn.Module):
    def __init__(self, n_mel_channels, n_frames_per_step,
                 encoder_embedding_dim, attention_dim,
                 attention_location_n_filters,
                 attention_location_kernel_size,
                 prenet_dim, decoder_rnn_dim,
                 max_decoder_steps, gate_threshold,
                 decoder_n_lstms, p_decoder_dropout):
        super(Decoder, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.decoder_n_lstms = decoder_n_lstms
        self.p_decoder_dropout = p_decoder_dropout

        self.prenet = Prenet(n_mel_channels, [prenet_dim, prenet_dim])

        self.lstm0 = nn.LSTMCell(prenet_dim + encoder_embedding_dim, decoder_rnn_dim)
        self.lstm1 = nn.LSTMCell(decoder_rnn_dim + encoder_embedding_dim, decoder_rnn_dim)

        self.attention_layer = Attention(decoder_rnn_dim, encoder_embedding_dim,
                                         attention_dim, attention_location_n_filters,
                                         attention_location_kernel_size)

        self.linear_projection = LinearNorm(decoder_rnn_dim + encoder_embedding_dim, n_mel_channels * n_frames_per_step)

        self.gate_layer = LinearNorm(decoder_rnn_dim + encoder_embedding_dim, n_frames_per_step, w_init_gain='sigmoid')

    def initialize_decoder_states(self, memory, mask=None):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.h0 = torch.zeros(B, self.decoder_rnn_dim).cuda()
        self.c0 = torch.zeros(B, self.decoder_rnn_dim).cuda()

        self.h1 = torch.zeros(B, self.decoder_rnn_dim).cuda()
        self.c1 = torch.zeros(B, self.decoder_rnn_dim).cuda()

        self.attention_weights = memory.new(B, MAX_TIME).zero_()
        self.attention_weights_cum = memory.new(B, MAX_TIME).zero_()
        self.attention_context = memory.new(B, self.encoder_embedding_dim).zero_()

        self.memory = memory
        self.mask = mask

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments, mel_lengths=None):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B, T_in) -> (B, T_out, T_in)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B, n_frames_per_step) -> (B, T_out, n_frames_per_step)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        # (B, T_out, n_frames_per_step) -> (B, T_out)
        gate_outputs = gate_outputs.contiguous().view(gate_outputs.size(0), -1)
        # (T_out, B, n_mel_channels * n_frames_per_step) -> (B, T_out, n_mel_channels * n_frames_per_step)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        # mel lengths scale to the target length
        if mel_lengths is not None:
            mel_lengths *= self.n_frames_per_step

        return mel_outputs, gate_outputs, alignments, mel_lengths

    def decode(self, prenet_output):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        prenet_output: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        x = torch.cat((prenet_output, self.attention_context), dim=-1)
        self.h0, self.c0 = self.lstm0(x, (self.h0, self.c0))
        x = F.dropout(self.h0, self.p_decoder_dropout, self.training)

        x = torch.cat((x, self.attention_context), dim=-1)
        self.h1, self.c1 = self.lstm1(x, (self.h1, self.c1))
        self.query = F.dropout(self.h1, self.p_decoder_dropout, self.training)

        attention_weights_cumulative = self.attention_weights_cum.unsqueeze(1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.query, self.memory, attention_weights_cumulative, self.mask)

        # [B, MAX_TIME]
        # Avoid '+=' as in-place operation in case of gradient computation
        self.attention_weights_cum = self.attention_weights_cum + self.attention_weights

        x = torch.cat((self.query, self.attention_context), dim=-1)
        # [B, n_mel_channels * n_frames_per_step]
        mel_output = self.linear_projection(x)
        # [B, n_frames_per_step]
        gate_output = self.gate_layer(x)
        return mel_output, gate_output, self.attention_weights

    def forward(self, memory, targets, memory_lengths, gta=False):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        targets: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        go_frame = memory.new(memory.size(0), self.n_mel_channels).zero_().unsqueeze(0)
        # (B, n_mel_channels, T_out) -> (T_out, B, n_mel_channels)
        targets = targets.permute(2, 0, 1)
        decoder_inputs = torch.cat((go_frame, targets), dim=0)
        prenet_outputs = self.prenet(decoder_inputs, inference=gta)

        mask =~ get_mask_from_lengths(memory_lengths) if memory.size(0) > 1 else None
        self.initialize_decoder_states(memory, mask)

        mel_outputs, gate_outputs, alignments = [], [], []
        # size - 1 for ignoring EOS synbol
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            prenet_output = prenet_outputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(prenet_output)

            mel_outputs += [mel_output]
            gate_outputs += [gate_output]
            alignments += [attention_weights]

        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

    def infer(self, memory, memory_lengths):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        frame = memory.new(memory.size(0), self.n_mel_channels).zero_()

        mask =~ get_mask_from_lengths(memory_lengths) if memory.size(0) > 1 else None
        self.initialize_decoder_states(memory, mask)

        mel_lengths = torch.zeros([memory.size(0)], dtype=torch.int32)
        if torch.cuda.is_available():
            mel_lengths = mel_lengths.cuda()

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            prenet_output = self.prenet(frame, inference=True)

            mel_output, gate_output, alignment = self.decode(prenet_output)
            gate_output = torch.sigmoid(gate_output)

            finished = torch.gt(gate_output, self.gate_threshold).all(-1)
            mel_lengths += (~finished).to(torch.int32)

            if finished.all():
                break

            mel_outputs += [mel_output]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            frame = mel_output[:, :self.n_mel_channels]

        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments, mel_lengths)


class Tacotron2(nn.Module):
    def __init__(self, mask_padding, n_mel_channels,
                 n_symbols, symbols_embedding_dim, encoder_kernel_size,
                 encoder_n_convolutions, encoder_embedding_dim,
                 attention_dim, attention_location_n_filters,
                 attention_location_kernel_size, n_frames_per_step,
                 prenet_dim, decoder_rnn_dim, max_decoder_steps, gate_threshold,
                 decoder_n_lstms, p_decoder_dropout,
                 postnet_embedding_dim, postnet_kernel_size,
                 postnet_n_convolutions):
        super(Tacotron2, self).__init__()
        self.elapsed_epochs = 0
        self.mask_padding = mask_padding
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        std = math.sqrt(2.0 / (n_symbols + symbols_embedding_dim))
        val = math.sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(encoder_n_convolutions, encoder_embedding_dim, encoder_kernel_size)
        self.decoder = Decoder(n_mel_channels, n_frames_per_step,
                               encoder_embedding_dim, attention_dim,
                               attention_location_n_filters,
                               attention_location_kernel_size,
                               prenet_dim, decoder_rnn_dim,
                               max_decoder_steps,
                               gate_threshold, decoder_n_lstms,
                               p_decoder_dropout)
        self.postnet = Postnet(n_mel_channels, postnet_embedding_dim, postnet_kernel_size, postnet_n_convolutions)

    def parse_outputs(self, outputs, target_lengths=None):
        if self.mask_padding and target_lengths is not None:
            mask = ~get_mask_from_lengths(target_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].masked_fill_(mask, 0.0)
            outputs[1].masked_fill_(mask, 0.0)
            outputs[2].masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return [output.cpu() for output in outputs]

    def forward(self, inputs, gta=False):
        texts, text_lengths, targets, target_lengths = inputs

        # [B, T_in] -> [B, embed_dim, T_in]
        embedded_inputs = self.embedding(texts).transpose(1, 2)
        # [B, T_in, encoder_dim]
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs_before, gate_outputs, alignments, _ = self.decoder(encoder_outputs, targets, text_lengths, gta)
        mel_outputs_after = mel_outputs_before + self.postnet(mel_outputs_before)

        return self.parse_outputs([mel_outputs_before, mel_outputs_after, gate_outputs, alignments])

    def infer(self, texts, text_lengths):
        # [B, T_in] -> [B, embed_dim, T_in]
        embedded_inputs = self.embedding(texts).transpose(1, 2)
        # [B, T_in, encoder_dim]
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs_before, gate_outputs, alignments, mel_lengths = self.decoder.infer(encoder_outputs, text_lengths)
        mel_outputs_after = mel_outputs_before + self.postnet(mel_outputs_before)

        return self.parse_outputs([mel_outputs_before, mel_outputs_after, gate_outputs, alignments, mel_lengths])

    def elapse_epoch(self):
        self.elapsed_epochs += 1

    def get_elapsed_epochs(self):
        return self.elapsed_epochs

    def restore_checkpoint(self, filepath):
        if os.path.exists(filepath) :
            def _checkpoint_from_distributed(state_dict):
                """
                Checks whether checkpoint was generated by DistributedDataParallel. DDP
                wraps model in additional "module.", it needs to be unwrapped for single
                GPU inference.
                :param state_dict: model's state dict
                """
                ret = False
                for key, _ in state_dict.items():
                    if key.find('module.') != -1:
                        ret = True
                        break
                return ret

            def _unwrap_distributed(state_dict):
                """
                Unwraps model from DistributedDataParallel.
                DDP wraps model in additional "module.", it needs to be removed for single
                GPU inference.
                :param state_dict: model's state dict
                """
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('module.', '')
                    new_state_dict[new_key] = value
                return new_state_dict

            print(f'Loading Weights: "{filepath}"')
            checkpoint = torch.load(filepath)
            self.elapsed_epochs = checkpoint['epoch']
            if _checkpoint_from_distributed(checkpoint['model']):
                checkpoint['model'] = _unwrap_distributed(checkpoint['model'])
            self.load_state_dict(checkpoint['model'])
        else:
            torch.save({'epoch': self.elapsed_epochs,
                        'model': self.state_dict()}, filepath)
