export CUDA_VISIBLE_DEVICES=0
python -m starvector.validation.validate \
config=configs/generation/hf/starvector-1b/text2svg.yaml \
dataset.name=/data/wdy/StarVector/text2svg-stack