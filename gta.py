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
from common.utils import load_metadata
from dllogger.logger import LOGGER
import dllogger.logger as dllg
from dllogger.autologging import log_hardware, log_args
from tqdm import tqdm


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-o', '--output', type=str, default='gta', help='output folder to save audio (file per phrase)')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int, help='Sampling rate')
    parser.add_argument('--checkpoint', type=str, default="logs/checkpoint_latest.pt", help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--dataset-path', type=str, default='filelists', help='Path to dataset')
    parser.add_argument('--anchor-dirs', default=['ljs_mel_text_train_filelist.txt'], type=str, nargs='*', help='Multi-speaker corpus directory')
    parser.add_argument('--text-cleaners', nargs='*', default=['basic_cleaners'], type=str, help='Type of text cleaners for input text')
    parser.add_argument('--amp-run', action='store_true', help='inference with AMP')
    parser.add_argument('--log-file', type=str, default='nvlog.json', help='Filename for logging')
    parser.add_argument('--include-warmup', action='store_true', help='Include warmup')
    parser.add_argument('--hop-length', type=int, default=256, help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('-r', '--reduction-factor', default=3, type=int, help='Number of frames processed per step')

    return parser


def load_checkpoint(checkpoint_path, model_name):
    assert os.path.isfile(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def load_and_setup_model(parser, args):
    checkpoint_path = args.checkpoint
    parser = parse_tacotron2_args(parser, add_help=False)
    args, _ = parser.parse_known_args()
    model = get_tacotron2_model(args, len(args.anchor_dirs), is_training=False)
    model.restore_checkpoint(checkpoint_path)
    model.eval()

    if args.amp_run:
        model, _ = amp.initialize(model, [], opt_level='O1')

    return model


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
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    LOGGER.set_model_name("Tacotron2_PyT")
    LOGGER.set_backends([
        dllg.StdOutBackend(log_file=None, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1),
        dllg.JsonBackend(log_file=args.log_file, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1)
    ])
    LOGGER.register_metric("tacotron2_frames_per_sec", metric_scope=dllg.TRAIN_ITER_SCOPE)
    LOGGER.register_metric("tacotron2_latency", metric_scope=dllg.TRAIN_ITER_SCOPE)
    LOGGER.register_metric("latency", metric_scope=dllg.TRAIN_ITER_SCOPE)

    model = load_and_setup_model(parser, args)

    log_hardware()
    log_args(args)

    if args.include_warmup:
        sequences = torch.randint(low=0, high=148, size=(1,50), dtype=torch.long).cuda()
        text_lengths = torch.IntTensor([sequence.size(1)]).cuda().long()
        for i in range(3):
            with torch.no_grad():
                outputs = model.infer(sequences, text_lengths)
                _, mels, _, _, mel_lengths = [output.cpu() for output in outputs]

    os.makedirs(args.output, exist_ok=True)

    LOGGER.iteration_start()

    measurements = {}

    anchor_dirs = [os.path.join(args.dataset_path, anchor) for anchor in args.anchor_dirs]
    metadatas = [load_metadata(anchor) for anchor in anchor_dirs]
    with torch.no_grad(), MeasureTime(measurements, "tacotron2_time"):
        for speaker_id in range(len(anchor_dirs)):
            metadata = metadatas[speaker_id]
            for mel_path, text in tqdm(metadata):
               seq = text_to_sequence(text, speaker_id, ['basic_cleaners'])
               seqs = torch.from_numpy(np.stack(seq)).unsqueeze(0)
               seq_lens = torch.IntTensor([len(text)])
               melspec = torch.from_numpy(np.load(mel_path))
               target = melspec[:, ::args.reduction_factor]
               targets = torch.from_numpy(np.stack(target)).unsqueeze(0)
               target_lengths = torch.IntTensor([target.shape[1]])
               inputs = (to_gpu(seqs).long(), to_gpu(seq_lens).int(), to_gpu(targets).float(), to_gpu(target_lengths).int())
               _, mel_outs, _, _ = model(inputs, gta=True)
               fname = os.path.basename(mel_path)
               np.save(os.path.join(args.output, fname), mel_outs[0, :, :melspec.shape[1]], allow_pickle=False)

    LOGGER.log(key="tacotron2_latency", value=measurements['tacotron2_time'])
    LOGGER.log(key="latency", value=(measurements['tacotron2_time']))
    LOGGER.iteration_stop()
    LOGGER.finish()


if __name__ == '__main__':
    main()
