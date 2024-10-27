from prompt_model import PromptTuningModel
from finetune_model import TraditionalFineTuningModel
from lora_model import LoRAModel
import torch.nn as nn
from typing import Optional, Tuple


def get_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function to create models"""
    models = {
        'prompt_tuning': PromptTuningModel,
        'traditional': TraditionalFineTuningModel,
        'lora': LoRAModel
    }
    
    if model_type not in models:
        raise ValueError(f"Model type {model_type} not supported. "
                        f"Choose from {list(models.keys())}")
    
    return models[model_type](**kwargs)



def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

