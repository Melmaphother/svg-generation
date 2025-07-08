#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
python starvector/validation/validator.py \
config=configs/generation/starvector-1b/im2svg.yaml \
dataset.name svg-stack \ 
model.generation_engine=hf

python starvector/validation/run_validator.py \
config=configs/generation/starvector-1b/im2svg.yaml \
dataset.name svg-emoji \ 
model.generation_engine=hf

python starvector/validation/run_validator.py \
config=configs/generation/starvector-1b/im2svg.yaml \
dataset.name svg-fonts \ 
model.generation_engine=hf

python starvector/validation/run_validator.py \
config=configs/generation/starvector-1b/im2svg.yaml \
dataset.name svg-diagrams \ 
model.generation_engine=hf

python starvector/validation/run_validator.py \
config=configs/generation/starvector-1b/im2svg.yaml \
dataset.name svg-icons \ 
model.generation_engine=hf



export CUDA_VISIBLE_DEVICES=7
python -m starvector.validation.validate \
config=configs/generation/hf/starvector-1b/im2svg.yaml \
dataset.name=/data/wdy/StarVector/svg-stack