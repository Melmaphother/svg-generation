export CUDA_VISIBLE_DEVICES=6
python -m starvector.validation.validate \
config=configs/generation/vllm-api/qwen2.5-vl-3b/text2svg_cogvlm.yaml \
dataset.dataset_name=/data/wdy/StarVector/text2svg-stack \
dataset.num_samples=-1 \
run.api.base_url=http://localhost:8010/v1
