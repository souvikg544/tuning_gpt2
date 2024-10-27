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

class PromptTuningModel(nn.Module):
    """Implementation of Prompt Tuning approach"""
    
    def __init__(self,
                 tokenizer,
                 model_name: str = 'gpt2',
                 num_virtual_tokens: int = 20
                 ):
        super().__init__()
        
        # Load the pretrained model
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = self.gpt2.config
        
        # Freeze the model parameters
        for param in self.gpt2.parameters():
            param.requires_grad = False
            
        # Initialize the virtual tokens
        self.num_virtual_tokens = num_virtual_tokens
        self.embedding_size = self.config.n_embd

        self.gpt2.resize_token_embeddings(len(tokenizer))
        summarize_token_id = tokenizer.convert_tokens_to_ids('[summarize]')
        summarize_embedding = self.gpt2.transformer.wte.weight[summarize_token_id].clone().detach()
        
        self.prompt_embeddings = nn.Parameter(
                summarize_embedding.unsqueeze(0).repeat(num_virtual_tokens, 1)
            )
            
        # Initialize dropout for prompt embeddings
        self.prompt_dropout = nn.Dropout(0.1)
        
    def forward(self, 
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None) -> ModelOutput:
        
        batch_size = input_ids.shape[0]
        
        # Create prompts for the batch
        prompts = self.prompt_dropout(self.prompt_embeddings)
        prompts = prompts.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Get input embeddings from the model
        inputs_embeds = self.gpt2.transformer.wte(input_ids)
        
        # Concatenate prompt embeddings with input embeddings
        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
        
        # Adjust attention mask for the added prompt tokens
        if attention_mask is not None:
            prompt_attention_mask = torch.ones(
                batch_size, self.num_virtual_tokens,
                device=attention_mask.device
            )
            attention_mask = torch.cat(
                (prompt_attention_mask, attention_mask), dim=1
            )
            
        # Adjust labels for the added prompt tokens
        if labels is not None:
            prompt_labels = torch.full(
                (batch_size, self.num_virtual_tokens),
                -100,  # Ignore prompt positions in loss calculation
                device=labels.device
            )
            labels = torch.cat((prompt_labels, labels), dim=1)
        
        # Forward pass through the model
        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return ModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states
        )
