# vllm https://docs.vllm.ai/en/v0.5.5/dev/sampling_params.html
# TODO: This is not maintained, need to update it to use the new VLLM API

from .svg_validator_base import SVGValidator, register_validator
from starvector.data.util import rasterize_svg, clean_svg, use_placeholder
from starvector.data.util import encode_image_base64
from svgpathtools import svgstr2paths
import os
import json
from copy import deepcopy
from openai import OpenAI
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from starvector.prompts.template import (
    QWEN_2_5_VL_3B_Image_2_SVG_V1,
    QWEN_2_5_VL_3B_Text_2_SVG_V1
)

@register_validator
class StarVectorVLLMAPIValidator(SVGValidator):
    def __init__(self, config):
        
        super().__init__(config)
        # Initialize VLLM OpenAI client here
        self.client = OpenAI(
            api_key=config.run.api.key,
            base_url=f"{config.run.api.base_url}",
        )
        if 'starvector-1b' in config.model.name:
            self.svg_end_token_id = 49154  # Adjust as needed
        elif 'starvector-8b' in config.model.name:
            self.svg_end_token_id = 49156  # Adjust as needed
        # else:
        #     self.tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        #     self.svg_end_token_id = self.tokenizer.encode("</svg>")[0]
        
        self.get_dataloader(config)
    
    def generate_svg(self, batch, generate_config):
        outputs = []
        for i, sample in enumerate(batch['Svg']):
            if self.task == "im2svg":
                image = rasterize_svg(sample, self.config.dataset.im_size)
                base64_image = encode_image_base64(image)
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": ""},
                ]
                system_prompt = QWEN_2_5_VL_3B_Image_2_SVG_V1
            else:
                content = [
                    # {"type": "text", "text": batch['caption'][i] + "<svg-start><svg "} # Conflict!
                    # caption_blip2, caption_cogvlm, caption_llava
                    {"type": "text", "text": batch['caption'][i]}
                ]
                system_prompt = QWEN_2_5_VL_3B_Text_2_SVG_V1

            response = self.client.chat.completions.create(
                model=self.config.model.vllm_served_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                temperature=generate_config['temperature'],
                top_p=generate_config['top_p'] if generate_config['num_beams'] == 1 else 1.0,
                max_tokens= generate_config['max_length'],
                frequency_penalty=generate_config['frequency_penalty'],
                presence_penalty=generate_config['presence_penalty'],
                # stop=[str(self.svg_end_token_id)],
                extra_body={
                    'use_beam_search': generate_config['num_beams'] > 1,
                    'best_of': generate_config['num_beams']
                },
                stream=generate_config['stream'],
                # logit_bias={self.svg_end_token_id: generate_config['logit_bias']} if generate_config['logit_bias'] else None,
            )
            
            if generate_config['stream']:
                generated_text = self._handle_stream_response(response)
            else:
                # generated_text = "<svg" + response.choices[0].message.content
                generated_text = response.choices[0].message.content
            
            outputs.append(generated_text)
        return outputs

    def get_dataloader(self, config):
        data = load_dataset(config.dataset.dataset_name, config.dataset.config_name, split=config.dataset.split)

        if config.dataset.num_samples != -1:
            data = data.select(range(config.dataset.num_samples))

        self.dataloader = DataLoader(data, batch_size=self.config.dataset.batch_size, shuffle=False, num_workers=self.config.dataset.num_workers)
    
    def release_memory(self):
        pass

    def _handle_stream_response(self, response):
        generated_text = "<svg"
        for chunk in response:
            new_text = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
            generated_text += new_text
        return generated_text
    