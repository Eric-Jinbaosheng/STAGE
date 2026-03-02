from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


def render_multimodal_prompt(processor: Any, prompt_text: str, add_generation_prompt: bool = True) -> str:
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return prompt_text


@dataclass
class SchemaTextSFTCollator:
    processor: Any
    max_length: int = 512
    add_generation_prompt: bool = False

    def _prompt_length(self, prompt_text: str, image: Any) -> int:
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        if "attention_mask" in prompt_inputs:
            return int(prompt_inputs["attention_mask"][0].sum().item())
        return int(prompt_inputs["input_ids"].shape[1])

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [
            render_multimodal_prompt(
                self.processor,
                str(x["prompt_text"]),
                add_generation_prompt=True,
            )
            for x in batch
        ]
        targets = [str(x["target_schema_text"]) for x in batch]
        full_texts = [p + t for p, t in zip(prompts, targets)]
        images = [x["image"] for x in batch]

        model_inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = model_inputs["input_ids"].clone()
        for i, (prompt_text, image) in enumerate(zip(prompts, images)):
            plen = self._prompt_length(prompt_text, image)
            plen = int(plen)
            labels[i, :plen] = -100
        if "attention_mask" in model_inputs:
            labels = labels.masked_fill(model_inputs["attention_mask"] == 0, -100)
        model_inputs["labels"] = labels

        model_inputs["sample_id"] = [x["sample_id"] for x in batch]
        model_inputs["prompt_text"] = prompts
        model_inputs["target_schema_text"] = targets
        model_inputs["instruction"] = [x["instruction"] for x in batch]
        model_inputs["instr_blank"] = [x.get("instr_blank", "") for x in batch]
        model_inputs["instr_shuffle"] = [x.get("instr_shuffle", "") for x in batch]
        model_inputs["instr_swap"] = [x.get("instr_swap", "") for x in batch]
        model_inputs["swap_valid"] = [bool(x.get("swap_valid", False)) for x in batch]
        return model_inputs
