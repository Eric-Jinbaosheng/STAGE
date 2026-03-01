from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class SchemaTextSFTCollator:
    processor: Any
    max_length: int = 512
    add_generation_prompt: bool = False

    def _tokenize_text(self, texts: List[str]):
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [str(x["prompt_text"]) for x in batch]
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
        prompt_tokens = self._tokenize_text(prompts)
        prompt_lens = prompt_tokens["attention_mask"].sum(dim=1).tolist()
        for i, plen in enumerate(prompt_lens):
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
