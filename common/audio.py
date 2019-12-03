import librosa
import librosa.filters
import numpy as np
from hparams import hparams
from scipy import signal
from scipy.io import wavfile


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

def dc_notch_filter(wav):
	# code from speex
	notch_radius = 0.982
	den = notch_radius ** 2 + 0.7 * (1 - notch_radius) ** 2
	b = np.array([1, -2, 1]) * notch_radius
	a = np.array([1, -2 * notch_radius, den])
	return signal.lfilter(b, a, wav)

def load_wav(path, sr):
	return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path):
	wav = dc_notch_filter(wav)
	wav = wav / np.abs(wav).max() * 0.999
	f1 = 0.5 * 32767 / max(0.01, np.max(np.abs(wav)))
	f2 = np.sign(wav) * np.power(np.abs(wav), 0.95)
	wav = f1 * f2
	#proposed by @dsmiller
	wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))

def preemphasis(wav, k):
	return signal.lfilter([1, -k], [1], wav)

def inv_preemphasis(wav, k):
	return signal.lfilter([1], [1, -k], wav)

def trim_silence(wav):
	'''Trim leading and trailing silence

	Useful for M-AILABS dataset if we choose to trim the extra 0.5 silence at beginning and end.
	'''
	#Thanks @begeekmyfriend and @lautjy for pointing out the params contradiction. These params are separate and tunable per dataset.
	return librosa.effects.trim(wav, top_db= hparams.trim_top_db, frame_length=hparams.trim_fft_size, hop_length=hparams.trim_hop_size)[0]

def get_hop_size():
	hop_size = hparams.hop_size
	if hop_size is None:
		assert hparams.frame_shift_ms is not None
		hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
	return hop_size

def linearspectrogram(wav):
	D = _stft(preemphasis(wav, hparams.preemphasis))
	S = _amp_to_db(np.abs(D)) - hparams.ref_level_db

	if hparams.signal_normalization:
		return _normalize(S)
	return S

def melspectrogram(wav):
	D = _stft(preemphasis(wav, hparams.preemphasis))
	S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db

	if hparams.signal_normalization:
		return _normalize(S)
	return S

def inv_linear_spectrogram(linear_spectrogram):
	'''Converts linear spectrogram to waveform using librosa'''
	if hparams.signal_normalization:
		D = _denormalize(linear_spectrogram)
	else:
		D = linear_spectrogram

	S = _db_to_amp(D + hparams.ref_level_db) #Convert back to linear
	return inv_preemphasis(_griffin_lim(S ** hparams.power), hparams.preemphasis)

def inv_mel_spectrogram(mel_spectrogram):
	'''Converts mel spectrogram to waveform using librosa'''
	if hparams.signal_normalization:
		D = _denormalize(mel_spectrogram)
	else:
		D = mel_spectrogram

	S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db))  # Convert back to linear
	return inv_preemphasis(_griffin_lim(S ** hparams.power), hparams.preemphasis)

def _griffin_lim(S):
	'''librosa implementation of Griffin-Lim
	Based on https://github.com/librosa/librosa/issues/434
	'''
	angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
	S_complex = np.abs(S).astype(np.complex)
	y = _istft(S_complex * angles)
	for i in range(hparams.griffin_lim_iters):
		angles = np.exp(1j * np.angle(_stft(y)))
		y = _istft(S_complex * angles)
	return y

def _stft(y):
	return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(), win_length=hparams.win_size)

def _istft(y):
	return librosa.istft(y, hop_length=get_hop_size(), win_length=hparams.win_size)

# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram):
	global _inv_mel_basis
	if _inv_mel_basis is None:
		_inv_mel_basis = np.linalg.pinv(_build_mel_basis())
	return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis():
	assert hparams.fmax <= hparams.sample_rate // 2
	return librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
							   fmin=hparams.fmin, fmax=hparams.fmax)

def _amp_to_db(x):
	min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
	return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
	return np.power(10.0, (x) * 0.05)

def _normalize(S):
	if hparams.allow_clipping_in_normalization:
		if hparams.symmetric_mels:
			return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
			 -hparams.max_abs_value, hparams.max_abs_value)
		else:
			return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)

	if hparams.symmetric_mels:
		return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
	else:
		return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))

def _denormalize(D):
	if hparams.allow_clipping_in_normalization:
		if hparams.symmetric_mels:
			return (((np.clip(D, -hparams.max_abs_value,
				hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
				+ hparams.min_level_db)
		else:
			return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

	if hparams.symmetric_mels:
		return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
	else:
		return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
