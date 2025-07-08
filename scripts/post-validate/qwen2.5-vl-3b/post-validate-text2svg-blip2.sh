export CUDA_VISIBLE_DEVICES=6
python -m starvector.validation.post_vali_calc_metrics \
config=configs/generation/vllm-api/qwen2.5-vl-3b/post_vali_text2svg_blip2.yaml \
dataset.dataset_name=/data/wdy/StarVector/text2svg-stack \
dataset.num_samples=200 \
run.date_time=\"20250708_192345\"
