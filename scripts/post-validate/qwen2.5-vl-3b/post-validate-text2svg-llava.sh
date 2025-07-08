export CUDA_VISIBLE_DEVICES=2
python -m starvector.validation.post_vali_calc_metrics \
config=configs/generation/vllm-api/qwen2.5-vl-3b/post_vali_text2svg_llava.yaml \
dataset.dataset_name=/data/wdy/StarVector/text2svg-stack