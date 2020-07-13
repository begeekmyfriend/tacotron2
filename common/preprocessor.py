from concurrent.futures import ProcessPoolExecutor
from functools import partial
from common import audio
import glob
import librosa
import numpy as np
import os


def build_from_path(hparams, input_dir, wav_dir, mel_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	futures = []
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	for root, _, files in os.walk(input_dir):
		for f in files:
			if f.endswith('.trn'):
				trn_file = os.path.join(root, f)
				with open(trn_file) as f:
					basename = trn_file[:-4]
					wav_file = basename + '.wav'
					basename = basename.split('/')[-1]
					text = f.readline().strip()
					futures.append(executor.submit(partial(_process_utterance, wav_dir, mel_dir, basename, wav_file, text, hparams)))

	return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(wav_dir, mel_dir, basename, wav_file, text, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- wav_dir: the directory to write the preprocessed wav into
		- mel_dir: the directory to write the mel spectograms into
		- basename: the basename of each file
		- wav_file: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, time_steps, mel_frames, text)
	"""
	try:
		# Load the audio as numpy array
		wav, sr = librosa.core.load(wav_file, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print(f'file {wav_file} present in csv metadata is not present in wav folder. skipping!')
		return None

	#rescale wav
	if hparams.rescale:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max

	#M-AILABS extra silence specific
	if hparams.trim_silence:
		wav = audio.trim_silence(wav)

	# Compute the mel scale spectrogram from the wav
	mel = audio.melspectrogram(wav).astype(np.float32)
	mel_frames = mel.shape[1]

	if mel_frames > hparams.max_mel_frames or len(text) > hparams.max_text_length:
		return None

	#Zero pad for quantized signal
	#time resolution adjustement
	#ensure length of raw audio is multiple of hop size so that we can use
	#transposed convolution to upsample
	r = mel_frames * audio.get_hop_size() - len(wav)
	wav = np.pad(wav, (0, r), mode='constant', constant_values=0.)
	assert len(wav) == mel_frames * audio.get_hop_size()
	time_steps = len(wav)

	# Write the spectrogram and audio to disk
	filename = f'{basename}.npy'
	np.save(os.path.join(wav_dir, filename), wav, allow_pickle=False)
	np.save(os.path.join(mel_dir, filename), mel, allow_pickle=False)

	# Return a tuple describing this training example
	return (filename, time_steps, mel_frames, text)
