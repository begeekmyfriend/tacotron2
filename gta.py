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

import argparse
import numpy as np
import os
import sys
import time
import torch
from apex import amp
from scipy.io.wavfile import write
from tacotron2.data_function import to_gpu
from tacotron2.loader import parse_tacotron2_args
from tacotron2.loader import get_tacotron2_model
from tacotron2.text import text_to_sequence
from train import parse_training_args
from common.audio_processing import griffin_lim
from common.layers import TacotronSTFT
from common.utils import load_metadata, load_wav_to_torch, save_wav
from dllogger.logger import LOGGER
import dllogger.logger as dllg
from dllogger.autologging import log_hardware, log_args
from tqdm import tqdm


def load_checkpoint(checkpoint_path, model_name):
    assert os.path.isfile(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def load_and_setup_model(parser, args):
    checkpoint_path = os.path.join('logs', args.latest_checkpoint_file)
    parser = parse_tacotron2_args(parser, add_help=False)
    args, _ = parser.parse_known_args()
    model = get_tacotron2_model(args, len(args.training_anchor_dirs), is_training=False)
    model.restore_checkpoint(checkpoint_path)
    model.eval()

    if args.amp_run:
        model, _ = amp.initialize(model, [], opt_level='O1')

    return model, args


# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(sequences):
    # Right zero-pad all one-hot text sequences to max input length
    text_lengths, ids_sorted_decreasing = torch.sort(
        torch.IntTensor([len(x) for x in sequences]),
        dim=0, descending=True)
    max_text_len = text_lengths[0]

    texts = []
    for i in range(len(ids_sorted_decreasing)):
        text = sequences[ids_sorted_decreasing[i]]
        texts.append(np.pad(text, [0, max_text_len - len(text)], mode='constant'))

    texts = torch.from_numpy(np.stack(texts))
    return texts, text_lengths, ids_sorted_decreasing


def prepare_input_sequence(texts, speaker_id):
    sequences = [text_to_sequence(text, speaker_id, ['basic_cleaners'])[:] for text in texts]
    texts, text_lengths, ids_sorted_decreasing = pad_sequences(sequences)

    if torch.cuda.is_available():
        texts = texts.cuda().long()
        text_lengths = text_lengths.cuda().int()
    else:
        texts = texts.long()
        text_lengths = text_lengths.int()

    return texts, text_lengths, ids_sorted_decreasing


class MeasureTime():
    def __init__(self, measurements, key):
        self.measurements = measurements
        self.key = key

    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.cuda.synchronize()
        self.measurements[self.key] = time.perf_counter() - self.t0


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU.
    """
    parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Inference')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    LOGGER.set_model_name("Tacotron2_PyT")
    LOGGER.set_backends([
        dllg.StdOutBackend(log_file=None, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1),
        dllg.JsonBackend(log_file=args.log_file, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1)
    ])
    LOGGER.register_metric("tacotron2_frames_per_sec", metric_scope=dllg.TRAIN_ITER_SCOPE)
    LOGGER.register_metric("tacotron2_latency", metric_scope=dllg.TRAIN_ITER_SCOPE)
    LOGGER.register_metric("latency", metric_scope=dllg.TRAIN_ITER_SCOPE)

    model, args = load_and_setup_model(parser, args)

    log_hardware()
    log_args(args)

    os.makedirs(args.output_dir, exist_ok=True)

    LOGGER.iteration_start()

    measurements = {}

    anchor_dirs = [os.path.join(args.dataset_path, anchor) for anchor in args.training_anchor_dirs]
    metadatas = [load_metadata(anchor) for anchor in anchor_dirs]
    stft = TacotronSTFT(args.filter_length, args.hop_length, args.win_length,
                        args.n_mel_channels, args.sampling_rate, args.mel_fmin, args.mel_fmax)
    with torch.no_grad(), MeasureTime(measurements, "tacotron2_time"):
        for speaker_id in range(len(anchor_dirs)):
            metadata = metadatas[speaker_id]
            for npy_path, text in tqdm(metadata):
                seq = text_to_sequence(text, speaker_id, ['basic_cleaners'])
                seqs = torch.from_numpy(np.stack(seq)).unsqueeze(0)
                seq_lens = torch.IntTensor([len(text)])
                wav = load_wav_to_torch(npy_path)
                mel = stft.mel_spectrogram(wav.unsqueeze(0))
                mel = mel.squeeze()
                max_target_len = mel.size(1) - 1
                max_target_len += args.n_frames_per_step - max_target_len % args.n_frames_per_step
                padded_mel = np.pad(mel, [(0, 0), (0, max_target_len - mel.size(1))], mode='constant', constant_values=args.mel_pad_val)
                target = padded_mel[:, ::args.n_frames_per_step]
                targets = torch.from_numpy(np.stack(target)).unsqueeze(0)
                target_lengths = torch.IntTensor([target.shape[1]])
                outputs = model.infer(to_gpu(seqs).long(), to_gpu(seq_lens).int(), to_gpu(targets).half(), to_gpu(target_lengths).int())
                _, mel_out, _, _ = [output.cpu() for output in outputs if output is not None]
                mel_out = mel_out.squeeze()[:, :mel.size(-1) - 1]
                assert(mel_out.shape[-1] == wav.shape[-1] // args.hop_length)
                fname = os.path.basename(npy_path)
                np.save(os.path.join(args.output_dir, fname), mel_out, allow_pickle=False)
                # GTA synthesis
                # magnitudes = stft.inv_mel_spectrogram(mel_out.squeeze())
                # wav = griffin_lim(magnitudes, stft.stft_fn, 60)
                # save_wav(wav, os.path.join(args.output_dir, 'eval.wav'))

    LOGGER.log(key="tacotron2_latency", value=measurements['tacotron2_time'])
    LOGGER.log(key="latency", value=(measurements['tacotron2_time']))
    LOGGER.iteration_stop()
    LOGGER.finish()


if __name__ == '__main__':
    main()
