import tensorflow as tf

# Default hyperparameters
hparams = tf.contrib.training.HParams(
	# Comma-separated list of cleaners to run on text prior to training and eval. For non-English
	# text, you may want to use "basic_cleaners" or "transliteration_cleaners".
	cleaners='basic_cleaners',

	#Hardware setup (TODO: multi-GPU parallel tacotron training)
	use_all_gpus = False, #Whether to use all GPU resources. If True, total number of available gpus will override num_gpus.
	num_gpus = 1, #Determines the number of gpus in use
	###########################################################################################################################################

	#Audio
	num_mels = 80, #Number of mel-spectrogram channels and local conditioning dimensionality
	rescale = False, #Whether to rescale audio prior to preprocessing
	rescaling_max = 0.999, #Rescaling value
	trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
	clip_mels_length = True, #For cases of OOM (Not really recommended, working on a workaround)
	max_mel_frames = 900,  #Only relevant when clip_mels_length = True
	max_text_length = 300,  #Only relevant when clip_mels_length = True
	sentence_span = 20,  # Number of mel hops for each sentence interval

	#Mel spectrogram
	n_fft = 1024, #Extra window size is filled with 0 paddings to match this parameter
	hop_size = 256, #For 22050Hz, 256 ~= 11.5 ms
	win_size = 1024, #For 22050Hz, 1024 ~= 46 ms (If None, win_size = n_fft)
	sample_rate = 22050, #22050 Hz (corresponding to ljspeech dataset)
	frame_shift_ms = None,
	preemphasis = 0.97, # preemphasis coefficient
	vocoder = 'wavernn', # Could be 'wavernn' or 'melgan'

	#Multi-speaker batch_size should be integer multiplies number of speakers.
	anchor_dirs = ['tts_fanfanli_22050', 'tts_xiaoya_22050', 'tts_yangluzhuo_22050', 'tts_yuanzhonglu_22050'],

	#M-AILABS (and other datasets) trim params
	trim_fft_size = 512,
	trim_hop_size = 128,
	trim_top_db = 60,

	#Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization = True,
	allow_clipping_in_normalization = False, #Only relevant if mel_normalization = True
	symmetric_mels = True, #Whether to scale the data to be symmetric around 0
	max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max]

	#Limits
	min_level_db = -120,
	ref_level_db = 20,
	fmin = 50, #Set this to 75 if your speaker is male! if female, 125 should help taking off noise. (To test depending on dataset)
	fmax = 7600,

	#Griffin Lim
	power = 1.2,
	griffin_lim_iters = 60,
	###########################################################################################################################################

	#Tacotron
	outputs_per_step = 2, #number of frames to generate at each decoding step (speeds up computation and allows for higher batch size)
	stop_at_any = False, #Determines whether the decoder should stop when predicting <stop> to any frame or to all of them
	batch_norm_position = 'after', #Can be in ('before', 'after'). Determines whether we use batch norm before or after the activation function (relu). Matter for debate.

	embedding_dim = 512, #dimension of embedding space

	enc_conv_num_layers = 3, #number of encoder convolutional layers
	enc_conv_kernel_size = (5, ), #size of encoder convolution filters for each layer
	enc_conv_channels = 512, #number of encoder convolutions filters for each layer
	encoder_lstm_units = 256, #number of lstm units for each direction (forward and backward)

	smoothing = False, #Whether to smooth the attention normalization function
	attention_dim = 128, #dimension of attention space
	attention_filters = 32, #number of attention convolution filters
	attention_kernel = (31, ), #kernel size of attention convolution
	cumulative_weights = True, #Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)

	#Attention synthesis constraints
	#"Monotonic" constraint forces the model to only look at the forwards attention_win_size steps.
	#"Window" allows the model to look at attention_win_size neighbors, both forward and backward steps.
	synthesis_constraint = False,  #Whether to use attention windows constraints in synthesis only (Useful for long utterances synthesis)
	synthesis_constraint_type = 'window', #can be in ('window', 'monotonic').
	attention_win_size = 7, #Side of the window. Current step does not count. If mode is window and attention_win_size is not pair, the 1 extra is provided to backward part of the window.

	prenet_layers = [256, 256], #number of layers and number of units of prenet
	decoder_layers = 2, #number of decoder lstm layers
	decoder_lstm_units = 1024, #number of decoder lstm units on each layer
	max_iters = 1000, #Max decoder steps during inference (Just for safety from infinite loop cases)

	postnet_num_layers = 5, #number of postnet convolutional layers
	postnet_kernel_size = (5, ), #size of postnet convolution filters for each layer
	postnet_channels = 512, #number of postnet convolution filters for each layer

	#Loss params
	mask_encoder = False, #whether to mask encoder padding while computing attention. Set to True for better prosody but slower convergence.
	mask_decoder = False, #Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not be weighted, else recommended pos_weight = 20)
	cross_entropy_pos_weight = 1, #Use class weights to reduce the stop token classes imbalance (by adding more penalty on False Negatives (FN)) (1 = disabled)
	###########################################################################################################################################

	#Tacotron Training
	#Reproduction seeds
	tacotron_random_seed = 5339, #Determines initial graph and operations (i.e: model) random state for reproducibility
	tacotron_data_random_state = 1234, #random state for train test split repeatability

	#performance parameters
	tacotron_swap_with_cpu = False, #Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

	#train/test split ratios, mini-batches sizes
	tacotron_batch_size = 36, #number of training samples on each training steps
	#Tacotron Batch synthesis supports ~16x the training batch size (no gradients during testing).
	#Training Tacotron with unmasked paddings makes it aware of them, which makes synthesis times different from training. We thus recommend masking the encoder.
	tacotron_synthesis_batch_size = 48, #DO NOT MAKE THIS BIGGER THAN 1 IF YOU DIDN'T TRAIN TACOTRON WITH "mask_encoder=True"!!
	tacotron_test_size = 0.05, #% of data to keep as test data, if None, tacotron_test_batches must be not None. (5% is enough to have a good idea about overfit)
	tacotron_test_batches = None, #number of test batches.

	#Learning rate schedule
	tacotron_decay_learning_rate = True, #boolean, determines if the learning rate will follow an exponential decay
	tacotron_start_decay = 40000, #Step at which learning decay starts
	tacotron_decay_steps = 40000, #Determines the learning rate decay slope (UNDER TEST)
	tacotron_decay_rate = 0.4, #learning rate decay rate (UNDER TEST)
	tacotron_initial_learning_rate = 1e-3, #starting learning rate
	tacotron_final_learning_rate = 1e-5, #minimal learning rate

	#Optimization parameters
	tacotron_adam_beta1 = 0.9, #AdamOptimizer beta1 parameter
	tacotron_adam_beta2 = 0.999, #AdamOptimizer beta2 parameter
	tacotron_adam_epsilon = 1e-6, #AdamOptimizer Epsilon parameter

	#Regularization parameters
	tacotron_reg_weight = 1e-6, #regularization weight (for L2 regularization)
	tacotron_scale_regularization = False, #Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)
	tacotron_zoneout_rate = 0.1, #zoneout rate for all LSTM cells in the network
	tacotron_dropout_rate = 0.5, #dropout rate for all convolutional layers + prenet
	tacotron_clip_gradients = True, #whether to clip gradients

	#Evaluation parameters
	tacotron_natural_eval = False, #Whether to use 100% natural eval (to evaluate Curriculum Learning performance) or with same teacher-forcing ratio as in training (just for overfit)

	#Decoder RNN learning can take be done in one of two ways:
	#       Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode='constant'
	#       Scheduled Sampling Scheme: From Teacher-Forcing to sampling from previous outputs is function of global step. (teacher forcing ratio decay) mode='scheduled'
	#The second approach is inspired by:
	#Bengio et al. 2015: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
	#Can be found under: https://arxiv.org/pdf/1506.03099.pdf
	tacotron_teacher_forcing_mode = 'constant', #Can be ('constant' or 'scheduled'). 'scheduled' mode applies a cosine teacher forcing ratio decay. (Preference: scheduled)
	tacotron_teacher_forcing_ratio = 1., #Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs, Only relevant if mode='constant'
	tacotron_teacher_forcing_init_ratio = 1., #initial teacher forcing ratio. Relevant if mode='scheduled'
	tacotron_teacher_forcing_final_ratio = 0., #final teacher forcing ratio. (Set None to use alpha instead) Relevant if mode='scheduled'
	tacotron_teacher_forcing_start_decay = 10000, #starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
	tacotron_teacher_forcing_decay_steps = 40000, #Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
	tacotron_teacher_forcing_decay_alpha = None, #teacher forcing ratio decay rate. Defines the final tfr as a ratio of initial tfr. Relevant if mode='scheduled'
	)

def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
	return 'Hyperparameters:\n' + '\n'.join(hp)
