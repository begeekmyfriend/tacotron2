# Tacotron 2 for PyTorch

This repository provides a script and recipe to train Tacotron 2. The source is forked from [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2) and combined with [Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2). It supports multi-speaker TTS, GTA synthesis as well as reduction factor. 

## Run the scripts
```shell
# Preprocessing
python preprocess.py
# Training
nohup bash scripts/train_tacotron2.sh &
# Evaluation
bash scripts/griffin_lim_synth.sh
# GTA synthesis
bash scripts/gta_synth.sh
```

## Vocoder recommended
[WaveRNN](https://github.com/begeekmyfriend/WaveRNN)

[WaveGlow](https://github.com/begeekmyfriend/WaveGlow)

[SqueezeWave](https://github.com/begeekmyfriend/SqueezeWave)

## Audio samples
[One male and one female](https://github.com/begeekmyfriend/tacotron2/issues/1)
