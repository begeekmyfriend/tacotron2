# *****************************************************************************
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

import random
import numpy as np
import os
import torch
import torch.utils.data
from common.layers import TacotronSTFT
from common.utils import load_wav_to_torch, load_meta_dataset
from tacotron2.text import text_to_sequence


class TextMelDataset(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, args, anchor_dirs):
        self.speaker_num = len(anchor_dirs)
        self.meta_dirs = [os.path.join(args.dataset_path, anchor_dirs[i]) for i in range(self.speaker_num)]
        self.metadatas = [load_meta_dataset(meta_dir) for meta_dir in self.meta_dirs]
        self.offsets = [-1] * self.speaker_num
        self.text_cleaners = args.text_cleaners
        self.max_wav_value = args.max_wav_value
        self.sampling_rate = args.sampling_rate
        # self.load_mel_from_disk = args.load_mel_from_disk
        self.stft = TacotronSTFT(args.filter_length, args.hop_length, args.win_length,
                                 args.n_mel_channels, args.sampling_rate, args.mel_fmin,
                                 args.mel_fmax)
        random.seed(1234)
        random.shuffle(self.metadatas)

    def get_mel_text_pair(self, speaker_id, metadata):
        npy_mel, text = metadata
        seq_len = len(text)
        seq = self.get_sequence(text, speaker_id)
        mel = self.get_mel(npy_mel)
        return (seq, mel, seq_len)

    def get_mel(self, filename):
        if False:#not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_sequence(self, text, speaker_id):
        return text_to_sequence(text, speaker_id, self.text_cleaners)

    def __getitem__(self, index):
        group = [self.get_mel_text_pair(i, self.metadatas[i][self.offsets[i]]) for i in range(self.speaker_num)]
        self.offsets = [(self.offsets[i] + 1) % len(self.metadatas[i]) for i in range(self.speaker_num)]
        return group

    def __len__(self):
        return sum([len(m) for m in self.metadatas]) // self.speaker_num


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, args):
        self.n_frames_per_step = args.n_frames_per_step
        self.mel_pad_val = args.mel_pad_val

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Flatten the batch
        batch = [sample for group in batch for sample in group]

        # Right zero-pad all one-hot text sequences to max input length
        text_lengths, ids_sorted_decreasing = torch.sort(
            torch.IntTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_text_len = text_lengths[0]

        texts = []
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            texts.append(np.pad(text, [0, max_text_len - len(text)], mode='constant'))

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        targets, reduced_targets = [], []
        gates = np.zeros([len(batch), max_target_len], dtype=np.float32)
        target_lengths = torch.IntTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            target_lengths[i] = mel.shape[1]
            gates[i, mel.shape[1] - 1:] = 1
            padded_mel = np.pad(mel, [(0, 0), (0, max_target_len - mel.size(1))], mode='constant', constant_values=self.mel_pad_val)
            targets.append(padded_mel)
            reduced_targets.append(padded_mel[:, ::self.n_frames_per_step])

        texts = torch.from_numpy(np.stack(texts))
        targets = torch.from_numpy(np.stack(targets))
        reduced_targets = torch.from_numpy(np.stack(reduced_targets))
        gates = torch.from_numpy(gates)
        return texts, text_lengths, targets, reduced_targets, gates, target_lengths, 


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def batch_to_gpu(batch):
    texts, text_lengths, targets, reduced_targets, gates, target_lengths = batch
    x = (to_gpu(texts).long(), to_gpu(text_lengths).int(), to_gpu(reduced_targets).float(), to_gpu(target_lengths).int())
    y = (targets, gates)
    num_frames = torch.sum(target_lengths)
    return (x, y, num_frames)
