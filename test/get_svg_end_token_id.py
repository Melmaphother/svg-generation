from transformers import AutoTokenizer

model_path_or_name = "/data/wdy/Downloads/models/Qwen/Qwen2.5-VL-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
svg_end_token_ids = tokenizer.encode("</svg>")
print(svg_end_token_ids)
for token_id in svg_end_token_ids:
    print(tokenizer.decode(token_id))