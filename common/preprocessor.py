import glob, os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from datasets import audio


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
		wav = audio.load_wav(wav_file, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print(f'file {wav_file} present in csv metadata is not present in wav folder. skipping!')
		return None

	#rescale wav
	if hparams.rescale:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max

	#M-AILABS extra silence specific
	if hparams.trim_silence:
		wav = audio.trim_silence(wav, hparams)

	#[-1, 1]
	out = encode_mu_law(wav, mu=512)

	# Compute the mel scale spectrogram from the wav
	mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
	mel_frames = mel_spectrogram.shape[1]

	if mel_frames > hparams.max_mel_frames or len(text) > hparams.max_text_length:
		return None

	#Zero pad for quantized signal
	#time resolution adjustement
	#ensure length of raw audio is multiple of hop size so that we can use
	#transposed convolution to upsample
	r = mel_frames * audio.get_hop_size(hparams) - len(wav)
	out = np.pad(out, (0, r), mode='constant', constant_values=0.)
	assert len(out) == mel_frames * audio.get_hop_size(hparams)
	time_steps = len(out)

	# Write the spectrogram and audio to disk
	filename = f'{basename}.npy'
	np.save(os.path.join(wav_dir, filename), out.astype(np.int16), allow_pickle=False)
	np.save(os.path.join(mel_dir, filename), mel_spectrogram, allow_pickle=False)

	# Return a tuple describing this training example
	return (filename, time_steps, mel_frames, text)


def label_2_float(x, bits) :
	return 2 * x / (2**bits - 1.) - 1.


def float_2_label(x, bits) :
	assert abs(x).max() <= 1.0
	x = (x + 1.) * (2**bits - 1) / 2
	return x.clip(0, 2**bits - 1)


def encode_mu_law(x, mu) :
	mu = mu - 1
	fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
	return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=False) :
	# TODO : get rid of log2 - makes no sense
	import math
	if from_labels : y = label_2_float(y, math.log2(mu))
	mu = mu - 1
	x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
	return x
