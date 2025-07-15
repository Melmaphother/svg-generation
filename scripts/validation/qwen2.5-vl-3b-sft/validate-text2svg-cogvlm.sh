export CUDA_VISIBLE_DEVICES=0
python -m starvector.validation.validate \
config=configs/generation/vllm-api/qwen2.5-vl-3b/text2svg_cogvlm.yaml \
run.api.base_url=http://localhost:8011/v1 \
model.vllm_served_name=qwen2_5vl_lora_sft_20k \
dataset.dataset_name=/data/wdy/StarVector/text2svg-stack \
dataset.num_samples=-1
