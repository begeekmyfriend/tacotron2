import argparse
import os
from multiprocessing import cpu_count

from datasets import preprocessor
from hparams import hparams
from tqdm import tqdm


def preprocess(args, input_folders, output_dir, hparams):
	mel_frames, timesteps = 0, 0
	max_text_lens, max_mel_lens, max_timestep_lens = [], [], []

	for input_dir in input_folders:
		wav_dir = os.path.join(output_dir, input_dir.split('/')[-1], 'audio')
		mel_dir = os.path.join(output_dir, input_dir.split('/')[-1], 'mels')
		os.makedirs(wav_dir, exist_ok=True)
		os.makedirs(mel_dir, exist_ok=True)
		metadata = preprocessor.build_from_path(hparams, input_dir, wav_dir, mel_dir, args.n_jobs, tqdm=tqdm)
		with open(os.path.join(output_dir, input_dir.split('/')[-1], 'train.txt'), 'w') as f:
			for m in metadata:
				f.write('|'.join([str(x) for x in m]) + '\n')
		max_text_lens.append(max(len(m[3]) for m in metadata))
		max_mel_lens.append(max(int(m[2]) for m in metadata))
		max_timestep_lens.append(max(m[1] for m in metadata))
		mel_frames += sum([int(m[2]) for m in metadata])
		timesteps += sum([int(m[1]) for m in metadata])

	hours = timesteps / hparams.sample_rate / 3600
	print(f'Write {len(metadata)} utterances, {mel_frames} mel frames, {timesteps} audio timesteps, ({hours:.2f} hours)')
	print(f'Max input length (text chars): {max(max_text_lens)}')
	print(f'Max mel frames length: {max(max_mel_lens)}')
	print(f'Max audio timesteps length: {max(max_timestep_lens)}')


def norm_data(args):

	merge_books = (args.merge_books=='True')

	print('Selecting data folders..')
	supported_datasets = ['LJSpeech-1.0', 'LJSpeech-1.1', 'M-AILABS', 'MANDARIN']
	if args.dataset not in supported_datasets:
		raise ValueError(f'dataset value entered {args.dataset} does not belong to supported datasets: {supported_datasets}')

	if args.dataset.startswith('LJSpeech'):
		return [os.path.join(args.base_dir, args.dataset)]

	if args.dataset.startswith('MANDARIN'):
		return [os.path.join(args.base_dir, 'data_thchs30', anchor) for anchor in hparams.anchor_dirs]

	if args.dataset == 'M-AILABS':
		supported_languages = ['en_US', 'en_UK', 'fr_FR', 'it_IT', 'de_DE', 'es_ES', 'ru_RU',
			'uk_UK', 'pl_PL', 'nl_NL', 'pt_PT', 'fi_FI', 'se_SE', 'tr_TR', 'ar_SA']
		if args.language not in supported_languages:
			raise ValueError(f'Please enter a supported language to use from M-AILABS dataset! \n{supported_languages}')

		supported_voices = ['female', 'male', 'mix']
		if args.voice not in supported_voices:
			raise ValueError(f'Please enter a supported voice option to use from M-AILABS dataset! \n{supported_voices}')

		path = os.path.join(args.base_dir, args.language, 'by_book', args.voice)
		supported_readers = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
		if args.reader not in supported_readers:
			raise ValueError(f'Please enter a valid reader for your language and voice settings! \n{supported_readers}')

		path = os.path.join(path, args.reader)
		supported_books = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
		if merge_books:
			return [os.path.join(path, book) for book in supported_books]

		else:
			if args.book not in supported_books:
				raise ValueError(f'Please enter a valid book for your reader settings! \n{supported_books}')
			return [os.path.join(path, args.book)]


def run_preprocess(args, hparams):
	input_folders = norm_data(args)
	output_folder = os.path.join(args.base_dir, args.output)
	preprocess(args, input_folders, output_folder, hparams)


def main():
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--dataset', default='MANDARIN')
	parser.add_argument('--language', default='en_US')
	parser.add_argument('--voice', default='female')
	parser.add_argument('--reader', default='mary_ann')
	parser.add_argument('--merge_books', default='False')
	parser.add_argument('--book', default='northandsouth')
	parser.add_argument('--output', default='training_data')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()

	modified_hp = hparams.parse(args.hparams)

	assert args.merge_books in ('False', 'True')

	run_preprocess(args, modified_hp)


if __name__ == '__main__':
	main()
