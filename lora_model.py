import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from peft import get_peft_model, LoraConfig, TaskType
import math
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class ModelOutput:
    """Custom dataclass for model outputs"""
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class LoRAModel(nn.Module):
    """Implementation of LoRA (Low-Rank Adaptation)"""
    
    def __init__(self,
                 tokenizer,
                 model_name: str = 'gpt2',
                 r: int = 8,
                 alpha: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        
        # Define LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["c_attn", "c_proj"]
        )
        
        # Load base model
        base_model = GPT2LMHeadModel.from_pretrained(model_name)
        base_model.resize_token_embeddings(len(tokenizer))
        
        # Create LoRA model
        self.model = get_peft_model(base_model, peft_config)
        
    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None) -> ModelOutput:
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return ModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states
        )
