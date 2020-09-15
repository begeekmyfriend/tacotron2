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

import numpy as np
from scipy.io.wavfile import read, write
from scipy import signal
import math
import torch
import os


def cosine_decay(init_val, final_val, step, decay_steps):
    alpha = final_val / init_val
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return init_val * decayed


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.IntTensor(max_len))
    mask = ids < lengths.unsqueeze(1)
    return mask


def preemphasize(wav, k=0.97):
    return signal.lfilter([1, -k], [1], wav)


def de_emphasize(wav, k=0.97):
    return signal.lfilter([1], [1, -k], wav)


def load_wav_to_torch(path, max_value=32768):
    wav = np.load(path)
    wav = preemphasize(wav)
    return torch.FloatTensor(wav.astype(np.float32))


def dc_notch_filter(wav):
    # code from speex
    notch_radius = 0.982
    den = notch_radius ** 2 + 0.7 * (1 - notch_radius) ** 2
    b = np.array([1, -2, 1]) * notch_radius
    a = np.array([1, -2 * notch_radius, den])
    return signal.lfilter(b, a, wav)


def save_wav(wav, path, sr=22050):
    wav = dc_notch_filter(wav)
    f1 = 0.8 * 32768 / max(0.01, np.max(np.abs(wav)))
    f2 = np.sign(wav) * np.power(np.abs(wav), 0.95)
    wav = f1 * f2
    write(path, sr, wav.astype(np.int16))


def load_metadata(dirname, filename='train.txt', split="|", load_mel_from_disk=False):
    with open(os.path.join(dirname, filename)) as f:
        def split_line(line):
            parts = line.strip().split(split)
            wav_path = os.path.join(dirname, 'mels' if load_mel_from_disk else 'audio', parts[0])
            text = parts[-1]
            return wav_path, text
        return [split_line(line) for line in f.readlines()]


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
