export CUDA_VISIBLE_DEVICES=1
python -m starvector.validation.validate \
config=configs/generation/vllm-api/qwen2.5-vl-3b/text2svg_blip2.yaml \
dataset.dataset_name=/data/wdy/StarVector/text2svg-stack \
dataset.num_samples=-1 \
run.api.base_url=http://localhost:8011/v1
