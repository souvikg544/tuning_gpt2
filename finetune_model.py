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

class TraditionalFineTuningModel(nn.Module):
    """Implementation of Traditional Fine-tuning (last layers only)"""
    
    def __init__(self,
                 tokenizer, 
                 model_name: str = 'gpt2',
                 num_layers_to_finetune: int = 2):
        super().__init__()
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))
        
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze the last n transformer layers
        for i in range(num_layers_to_finetune):
            layer_idx = self.model.config.n_layer - i - 1
            for param in self.model.transformer.h[layer_idx].parameters():
                param.requires_grad = True
                
        # Unfreeze the output layer and layer norm
        for param in self.model.lm_head.parameters():
            param.requires_grad = True
        for param in self.model.transformer.ln_f.parameters():
            param.requires_grad = True
            
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
        # print(outputs)
        # return outputs

        return ModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states
        )
