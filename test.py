import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
from typing import Optional, Tuple
from get_model import get_model
import gc


def create_dummy_data(num_samples: int = 100, seq_length: int = 128, vocab_size: int = 50257):
    """Create dummy data for testing"""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, vocab_size, (num_samples, seq_length))
    return TensorDataset(input_ids, attention_mask, labels)

def check_gradients(model: nn.Module) -> dict:
    """Check if gradients are properly flowing through the model"""
    grad_status = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_status[name] = {
                'requires_grad': param.requires_grad,
                'grad_exists': param.grad is not None,
                'grad_norm': torch.norm(param.grad).item() if param.grad is not None else None
            }
    return grad_status

def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                gradient_check_step: int = 10) -> tuple:
    """Train for one epoch and return losses and gradient information"""
    model.train()
    total_loss = 0
    all_losses = []
    gradient_info = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    for step, (input_ids, attention_mask, labels) in enumerate(progress_bar):
        # Move batch to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Backward pass
        outputs.loss.backward()
        
        # Check gradients periodically
        if step % gradient_check_step == 0:
            grad_info = check_gradients(model)
            gradient_info.append(grad_info)
        
        # Update weights
        optimizer.step()
        
        # Track loss
        total_loss += outputs.loss.item()
        all_losses.append(outputs.loss.item())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': outputs.loss.item()})
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_losses, gradient_info

def plot_training_loss(losses: list, title: str, save_path: Optional[str] = None):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(f'Training Loss - {title}')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def test_training(model_type: str, config: dict, num_epochs: int = 2):
    """Test training for a specific model type"""
    print(f"\nTesting training for {model_type} model...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create model
        model = get_model(model_type, **config)
        model = model.to(device)
        
        # Create dummy dataset
        dataset = create_dummy_data()
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Setup optimizer
        optimizer = AdamW(model.parameters(), lr=5e-5)
        
        # Training loop
        all_losses = []
        all_grad_info = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            avg_loss, epoch_losses, grad_info = train_epoch(
                model, dataloader, optimizer, device
            )
            
            all_losses.extend(epoch_losses)
            all_grad_info.extend(grad_info)
            
            print(f"Average loss: {avg_loss:.4f}")
            
            # Check gradient flow
            print("\nGradient flow check:")
            for param_name, grad_data in all_grad_info[-1].items():
                if grad_data['grad_exists']:
                    print(f"{param_name}:")
                    print(f"  Gradient norm: {grad_data['grad_norm']:.4f}")
        
        # Plot loss curve
        plot_training_loss(all_losses, f"{model_type} Training")
        
        return all_losses, all_grad_info
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e
    
    finally:
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Test training for all model types"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    model_configs = {
        # 'prompt_tuning': {'num_virtual_tokens': 20},
        # 'traditional': {'num_layers_to_finetune': 2},
        'lora': {'tokenizer':tokenizer,'r': 8, 'alpha': 16}
    }
    
    results = {}
    
    for model_type, config in model_configs.items():
        try:
            losses, grad_info = test_training(model_type, config)
            results[model_type] = {
                'losses': losses,
                'grad_info': grad_info
            }
            
            # Print final gradient stats
            print(f"\nFinal gradient statistics for {model_type}:")
            final_grads = grad_info[-1]
            trainable_params = [name for name, info in final_grads.items() 
                              if info['requires_grad']]
            params_with_grads = [name for name, info in final_grads.items() 
                               if info['grad_exists']]
            
            print(f"Trainable parameters: {len(trainable_params)}")
            print(f"Parameters with gradients: {len(params_with_grads)}")
            
            if len(trainable_params) != len(params_with_grads):
                print("WARNING: Some trainable parameters are not receiving gradients!")
                missing_grads = set(trainable_params) - set(params_with_grads)
                print(f"Parameters missing gradients: {missing_grads}")
            
        except Exception as e:
            print(f"Failed to test {model_type}: {str(e)}")
    
    return results

if __name__ == "__main__":
    results = main()