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

import os
import time
import argparse
import numpy as np
from common.utils import cosine_decay
from contextlib import contextmanager
from datetime import datetime
from plot import plot_alignment
import torch
from torch.utils.data import DataLoader

import torch.distributed as dist

#from apex.parallel import DistributedDataParallel as DDP

from tacotron2.loader import parse_tacotron2_args
from tacotron2.loader import get_tacotron2_model
from tacotron2.loss_function import Tacotron2Loss
from tacotron2.data_function import TextMelCollate
from tacotron2.data_function import TextMelLoader
from tacotron2.data_function import batch_to_gpu
from dllogger.logger import LOGGER
import dllogger.logger as dllg
from dllogger import tags
from dllogger.autologging import log_hardware, log_args
from scipy.io.wavfile import write as write_wav

# from apex import amp
# amp.lists.functional_overrides.FP32_FUNCS.remove('softmax')
# amp.lists.functional_overrides.FP16_FUNCS.append('softmax')


def parse_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_directory', type=str, default='logs', required=True, help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='training_data/mels', help='Path to dataset')
    parser.add_argument('--log-file', type=str, default='nvlog.json', help='Filename for logging')
    parser.add_argument('--latest-checkpoint-file', type=str, default='checkpoint_latest.pt', help='Store the latest checkpoint in each epoch')
    parser.add_argument('--phrase-path', type=str, default=None, help='Path to phrase sequence file used for sample generation')
    parser.add_argument('--tacotron2-checkpoint', type=str, default=None, help='Path to pre-trained Tacotron2 checkpoint for sample generation')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, required=True, help='Number of total epochs to run')
    training.add_argument('--epochs-per-alignment', type=int, default=10, help='Number of epochs per alignment')
    training.add_argument('--epochs-per-checkpoint', type=int, default=50, help='Number of epochs per checkpoint')
    training.add_argument('--seed', type=int, default=1234, help='Seed for PyTorch random number generators')
    training.add_argument('--dynamic-loss-scaling', type=bool, default=True, help='Enable dynamic loss scaling')
    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', action='store_true', help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', action='store_true', help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true', help='disable uniform initialization of batchnorm layer weight')

    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument('--use-saved-learning-rate', default=False, type=bool)
    optimization.add_argument('--init-lr', '--initial-learning-rate', default=1e-3, type=float, required=True, help='Initial learing rate')
    optimization.add_argument('--final-lr', '--final-learning-rate', default=1e-5, type=float, required=True, help='Final earing rate')
    optimization.add_argument('--weight-decay', default=1e-6, type=float, help='Weight decay')
    optimization.add_argument('--grad-clip-thresh', default=1.0, type=float, help='Clip threshold for gradients')
    optimization.add_argument('-bs', '--batch-size', default=32, type=int, required=True, help='Batch size per GPU')

    # dataset parameters
    dataset = parser.add_argument_group('dataset parameters')
    dataset.add_argument('--load-mel-from-disk', action='store_true', help='Loads mel spectrograms from disk instead of computing them on the fly')
    dataset.add_argument('--training-files', default='filelists/xiaoya_audio_text_train_filelist.txt', type=str, help='Path to training filelist')
    dataset.add_argument('--validation-files', default='filelists/xiaoya_audio_text_val_filelist.txt', type=str, help='Path to validation filelist')
    dataset.add_argument('--text-cleaners', nargs='*', default=['basic_cleaners'], type=str, help='Type of text cleaners for input text')

    # audio parameters
    audio = parser.add_argument_group('audio parameters')
    audio.add_argument('--max-wav-value', default=32768.0, type=float, help='Maximum audiowave value')
    audio.add_argument('--sampling-rate', default=22050, type=int, help='Sampling rate')
    audio.add_argument('--filter-length', default=2048, type=int, help='Filter length')
    audio.add_argument('--hop-length', default=275, type=int, help='Hop (stride) length')
    audio.add_argument('--win-length', default=1100, type=int, help='Window length')
    audio.add_argument('--mel-fmin', default=125.0, type=float, help='Minimum mel frequency')
    audio.add_argument('--mel-fmax', default=7600.0, type=float, help='Maximum mel frequency')

    distributed = parser.add_argument_group('distributed setup')
    # distributed.add_argument('--distributed-run', default=True, type=bool, help='enable distributed run')
    distributed.add_argument('--rank', default=0, type=int, help='Rank of the process, do not set! Done by multiproc module')
    distributed.add_argument('--world-size', default=1, type=int, help='Number of processes, do not set! Done by multiproc module')
    distributed.add_argument('--dist-url', type=str, default='tcp://localhost:23456', help='Url used to set up distributed training')
    distributed.add_argument('--group-name', type=str, default='group_name', required=False, help='Distributed group name')
    distributed.add_argument('--dist-backend', default='nccl', type=str, choices={'nccl'}, help='Distributed run backend')

    return parser


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= num_gpus
    return rt


def init_distributed(args, world_size, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=world_size, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def save_eval(model, filepath, args):
    if args.phrase_path:
        phrase = torch.load(args.phrase_path, map_location='cpu')
        with torch.no_grad():
            model.eval()
            mel = model.infer(phrase.cuda())[0].cpu()
            print(mel.size())
            model.train()

    # audio = audio[0].numpy()
    # audio = audio.astype('int16')
    # write_wav(filepath, sampling_rate, audio)

# adapted from: https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
# Following snippet is licensed under MIT license


@contextmanager
def evaluating(model):
    '''Temporarily switch to evaluation mode.'''
    try:
        model.eval()
        yield model
    finally:
        if model.training:
            model.train()


def validate(model, criterion, valset, iteration, collate_fn, distributed_run, args):
    """Handles all the validation scoring and printing"""
    with evaluating(model), torch.no_grad():
        val_loader = DataLoader(valset, num_workers=1, shuffle=False,
                                batch_size=args.batch_size, pin_memory=False,
                                collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y, num_frames = batch_to_gpu(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, args.world_size).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    LOGGER.log(key="val_iter_loss", value=reduced_val_loss)


def adjust_learning_rate(epoch, optimizer, args):
    lr = cosine_decay(args.init_lr, args.final_lr, epoch, args.epochs)

    if optimizer.param_groups[0]['lr'] != lr:
        LOGGER.log_event("learning_rate changed",
                         value=str(optimizer.param_groups[0]['lr']) + " -> " + str(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Training')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    LOGGER.set_model_name("Tacotron2_PyT")
    LOGGER.set_backends([
        dllg.StdOutBackend(log_file=None, logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1),
        dllg.JsonBackend(log_file=os.path.join(args.output_directory, args.log_file) if args.rank == 0 else None,
                         logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1)
    ])

    LOGGER.timed_block_start("run")
    LOGGER.register_metric(tags.TRAIN_ITERATION_LOSS, metric_scope=dllg.TRAIN_ITER_SCOPE)
    LOGGER.register_metric("iter_time", metric_scope=dllg.TRAIN_ITER_SCOPE)
    LOGGER.register_metric("epoch_time", metric_scope=dllg.EPOCH_SCOPE)
    LOGGER.register_metric("run_time", metric_scope=dllg.RUN_SCOPE)
    LOGGER.register_metric("val_iter_loss", metric_scope=dllg.EPOCH_SCOPE)
    LOGGER.register_metric("train_epoch_frames/sec", metric_scope=dllg.EPOCH_SCOPE)
    LOGGER.register_metric("train_epoch_avg_frames/sec", metric_scope=dllg.EPOCH_SCOPE)
    LOGGER.register_metric("train_epoch_avg_loss", metric_scope=dllg.EPOCH_SCOPE)

    log_hardware()

    parser = parse_tacotron2_args(parser)
    args = parser.parse_args()

    log_args(args)

    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    distributed_run = args.world_size > 1
    if distributed_run:
        init_distributed(args, args.world_size, args.rank, args.group_name)

    os.makedirs(args.output_directory, exist_ok=True)

    LOGGER.log(key=tags.RUN_START)
    run_start_time = time.time()

    model = get_tacotron2_model(args, is_training=True)

    if not args.amp_run and distributed_run:
        model = DDP(model)

    model.restore_checkpoint(os.path.join(args.output_directory, args.latest_checkpoint_file))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)

    if args.amp_run:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        if distributed_run:
            model = DDP(model)

    criterion = Tacotron2Loss()

    collate_fn = TextMelCollate(args)
    trainset = TextMelLoader(args, args.training_files)
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              batch_size=args.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)

    valset = TextMelLoader(args, args.validation_files)

    model.train()

    elapsed_epochs = model.get_elapsed_epochs()
    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)

    LOGGER.log(key=tags.TRAIN_LOOP)

    for epoch in range(1, epochs + 1):
        LOGGER.epoch_start()
        epoch_start_time = time.time()
        epoch += elapsed_epochs
        LOGGER.log(key=tags.TRAIN_EPOCH_START, value=epoch)

        # used to calculate avg frames/sec over epoch
        reduced_num_frames_epoch = 0

        # used to calculate avg loss over epoch
        train_epoch_avg_loss = 0.0
        train_epoch_avg_frames_per_sec = 0.0
        num_iters = 0

        for i, batch in enumerate(train_loader):
            print(f"Batch: {i}/{len(train_loader)} epoch {epoch}")
            LOGGER.iteration_start()
            iter_start_time = time.time()
            LOGGER.log(key=tags.TRAIN_ITER_START, value=i)

            # start = time.perf_counter()
            adjust_learning_rate(epoch, optimizer, args)

            optimizer.zero_grad()
            x, y, num_frames = batch_to_gpu(batch)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            if distributed_run:
                reduced_loss = reduce_tensor(loss.data, args.world_size).item()
                reduced_num_frames = reduce_tensor(num_frames.data, 1).item()
            else:
                reduced_loss = loss.item()
                reduced_num_frames = num_frames.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            LOGGER.log(key=tags.TRAIN_ITERATION_LOSS, value=reduced_loss)

            train_epoch_avg_loss += reduced_loss
            num_iters += 1

            # accumulate number of frames processed in this epoch
            reduced_num_frames_epoch += reduced_num_frames

            if args.amp_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.grad_clip_thresh)
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)

            optimizer.step()

            iteration += 1

            LOGGER.log(key=tags.TRAIN_ITER_STOP, value=i)

            iter_stop_time = time.time()
            iter_time = iter_stop_time - iter_start_time
            frames_per_sec = reduced_num_frames/iter_time
            train_epoch_avg_frames_per_sec += frames_per_sec

            LOGGER.log(key="train_iter_frames/sec", value=frames_per_sec)
            LOGGER.log(key="iter_time", value=iter_time)
            LOGGER.iteration_stop()

        LOGGER.log(key=tags.TRAIN_EPOCH_STOP, value=epoch)
        epoch_stop_time = time.time()
        epoch_time = epoch_stop_time - epoch_start_time

        LOGGER.log(key="train_epoch_frames/sec", value=(reduced_num_frames_epoch/epoch_time))
        LOGGER.log(key="train_epoch_avg_frames/sec", value=(train_epoch_avg_frames_per_sec/num_iters if num_iters > 0 else 0.0))
        LOGGER.log(key="train_epoch_avg_loss", value=(train_epoch_avg_loss/num_iters if num_iters > 0 else 0.0))
        LOGGER.log(key="epoch_time", value=epoch_time)

        LOGGER.log(key=tags.EVAL_START, value=epoch)

        # validate(model, criterion, valset, iteration, collate_fn, distributed_run, args)

        LOGGER.log(key=tags.EVAL_STOP, value=epoch)

        # Store latest checkpoint in each epoch
        model.elapse_epoch()
        checkpoint_path = os.path.join(args.output_directory, args.latest_checkpoint_file)
        torch.save(model.state_dict(), checkpoint_path)

        # Plot alignemnt
        if epoch % args.epochs_per_alignment == 0 and args.rank == 0:
            target_max_length = y[0].size(-1)
            alignments = y_pred[3].data.numpy()
            plot_alignment(alignments[0].transpose(0, 1),
                           os.path.join(args.output_directory, f"align_{epoch:04d}_{iteration}.png"),
                           info=f"{datetime.now().strftime('%Y-%m-%d %H:%M')} Epoch={epoch:04d} Iteration={iteration} Average loss={train_epoch_avg_loss/num_iters:.5f}",
                           max_len=target_max_length//args.n_frames_per_step)

        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0 and args.rank == 0:
            checkpoint_path = os.path.join(args.output_directory, f"checkpoint_{epoch:04d}.pt")
            print(f"Saving model and optimizer state at epoch {epoch:04d} to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)

            # Save evaluation
            # save_sample(model, args.tacotron2_checkpoint, args.phrase_path,
            #             os.path.join(args.output_directory, f"sample_{epoch:04d}_{iteration}.wav"), args.sampling_rate)

        LOGGER.epoch_stop()

    run_stop_time = time.time()
    run_time = run_stop_time - run_start_time
    LOGGER.log(key="run_time", value=run_time)
    LOGGER.log(key=tags.RUN_FINAL)

    print("training time", run_stop_time - run_start_time)

    LOGGER.timed_block_stop("run")

    if args.rank == 0:
        LOGGER.finish()


if __name__ == '__main__':
    main()
