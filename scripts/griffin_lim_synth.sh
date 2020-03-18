CUDA_VISIBLE_DEVICES=0 python inference.py -i text.txt -o outputs --amp-run --speaker-num 4 --speaker-id 0 --log-file nvlog.json
