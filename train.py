import os
import argparse
import torch
import wandb
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Dict, Any, Optional
from dataset import SummarizationDataset, create_dataloader
from transformers import GPT2Tokenizer
from get_model import get_model  # Your model loading function
import logging
from pathlib import Path
import json
from rouge_score import rouge_scorer

class Trainer:
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        self.setup_device()
        self.setup_tokenizer()
        self.setup_data()
        self.setup_model()
        self.setup_optimization()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        if args.wandb:
            self.setup_wandb()

    def setup_logging(self):
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'Using device: {self.device}')

    def setup_tokenizer(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Add special tokens if needed
        if self.args.training_strategy == 'prompt_tuning':
            special_tokens = {
            'sep_token': '<|sep|>',
            'pad_token': '<|pad|>',
            'bos_token': '<|startoftext|>',
            'eos_token': '<|endoftext|>',
            'summary_token' : '[summarize]'
        }
        else:
            special_tokens = {
            'sep_token': '<|sep|>',
            'pad_token': '<|pad|>',
            'bos_token': '<|startoftext|>',
            'eos_token': '<|endoftext|>'
            }
        self.tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})

    def setup_data(self):
        # Create datasets
        self.train_dataset = SummarizationDataset(
            self.args.train_data_path,
            self.tokenizer,
            max_article_length=self.args.max_article_length,
            max_summary_length=self.args.max_summary_length
        )
        
        self.val_dataset = SummarizationDataset(
            self.args.val_data_path,
            self.tokenizer,
            max_article_length=self.args.max_article_length,
            max_summary_length=self.args.max_summary_length
        )
        
        # Create dataloaders
        self.train_dataloader = create_dataloader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )
        
        self.val_dataloader = create_dataloader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False
        )

    def setup_model(self):
        # Model configuration based on training strategy
        config = {
            'prompt_tuning': {'tokenizer': self.tokenizer, 'num_virtual_tokens': self.args.num_virtual_tokens},
            'traditional': {'tokenizer': self.tokenizer, 'num_layers_to_finetune': self.args.num_layers_to_finetune},
            'lora': {'tokenizer': self.tokenizer, 'r': self.args.lora_r, 'alpha': self.args.lora_alpha}
        }
        
        self.model = get_model(
            self.args.training_strategy,
            **config[self.args.training_strategy]
        ).to(self.device)



    def setup_optimization(self):
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Learning rate scheduler
        num_training_steps = len(self.train_dataloader) * self.args.num_epochs
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def setup_wandb(self):
        wandb.init(
            project=self.args.wandb_project,
            name=f"{self.args.training_strategy}-{self.args.run_name}",
            config=vars(self.args)
        )

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc='Training')
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            
            # Backward pass
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += outputs.loss.item()
            progress_bar.set_postfix({'loss': outputs.loss.item()})
            
        return {'loss': total_loss / len(self.train_dataloader)}

    def generate_summary(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                        max_length: int) -> torch.Tensor:
        """
        Custom generation function using greedy decoding
        """
        self.model.eval()
        batch_size = input_ids.shape[0]
        
        # Find the separator token position for each sequence in the batch
        sep_token_id = self.tokenizer.convert_tokens_to_ids('<|sep|>')
        eos_token_id = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
        
        # Initialize generation with input up to separator token
        generated = []
        
        for i in range(batch_size):
            # Find position of separator token
            sep_pos = (input_ids[i] == sep_token_id).nonzero().item()
            
            # Initialize current sequence (everything up to separator token)
            curr_seq = input_ids[i,:sep_pos+1].unsqueeze(0)
            curr_mask = attention_mask[i,:sep_pos+1].unsqueeze(0)
            
            # Generate tokens one by one
            for _ in range(max_length):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=curr_seq,
                        attention_mask=curr_mask
                    )
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                # Stop if we predict the EOS token
                if next_token.item() == eos_token_id:
                    break
                    
                # Append predicted token
                print(curr_seq.shape,next_token.shape,next_token.unsqueeze(0).shape)
                curr_seq = torch.cat([curr_seq, next_token.unsqueeze(0)], dim=1)
                curr_mask = torch.cat([curr_mask, torch.ones(1, 1, device=self.device)], dim=1)
            
            generated.append(curr_seq.squeeze(0))
            
        # Pad sequences to same length
        max_len = max(seq.size(0) for seq in generated)
        padded_generated = []
        
        for seq in generated:
            padding_length = max_len - seq.size(0)
            padded_seq = torch.cat([
                seq,
                torch.full((padding_length,), self.tokenizer.convert_tokens_to_ids('<|pad|>'), device=self.device)
            ])
            padded_generated.append(padded_seq)
            
        return torch.stack(padded_generated)

    def evaluate(self,num_samples = 20) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_rouge_scores = []
        i =0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc='Evaluating'):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get model outputs for loss calculation
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                
                # Generate summaries using our custom generation function
                generated = self.generate_summary(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=self.args.max_summary_length
                )
                
                # Calculate ROUGE scores
                for gen, label in zip(generated, batch['labels']):
                    # Decode generated summary
                    gen_text = self.tokenizer.decode(
                        [t for t in gen if t != self.tokenizer.pad_token_id],
                        skip_special_tokens=True
                    )
                    
                    # Decode reference summary
                    label_text = self.tokenizer.decode(
                        [t for t in label if t != -100],
                        skip_special_tokens=True
                    )
                    
                    # Calculate scores
                    scores = self.scorer.score(label_text, gen_text)
                    all_rouge_scores.append(scores)
                i+=1
                if num_samples == i:
                    break
        
        # Calculate average scores
        avg_scores = {
            'val_loss': total_loss / num_samples,
            'rouge1': np.mean([s['rouge1'].fmeasure for s in all_rouge_scores]),
            'rouge2': np.mean([s['rouge2'].fmeasure for s in all_rouge_scores]),
            'rougeL': np.mean([s['rougeL'].fmeasure for s in all_rouge_scores])
        }
        
        return avg_scores

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        checkpoint_path = self.output_dir / f'checkpoint-epoch-{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'Saved checkpoint to {checkpoint_path}')

    def train(self):
        best_rouge = 0
        for epoch in range(self.args.num_epochs):
            self.logger.info(f'\nEpoch {epoch + 1}/{self.args.num_epochs}')
            
            # Training
            train_metrics = self.train_epoch()
            self.logger.info(f'Training metrics: {train_metrics}')
            
            # Evaluation
            val_metrics = self.evaluate()
            self.logger.info(f'Validation metrics: {val_metrics}')
            
            # Logging
            if self.args.wandb:
                wandb.log({**train_metrics, **val_metrics})
            
            # Save checkpoint if best model
            if val_metrics['rouge2'] > best_rouge:
                best_rouge = val_metrics['rouge2']
                self.save_checkpoint(epoch, val_metrics)

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT-2 for summarization')
    
    # Data arguments
    parser.add_argument('--train_data_path', type=str, default='/ssd_scratch/cvit/souvik/cnn_dailymail/train_small.csv', required=True)
    parser.add_argument('--val_data_path', type=str, default='/ssd_scratch/cvit/souvik/cnn_dailymail/val_small.csv', required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    
    # Model arguments
    parser.add_argument('--training_strategy', type=str, 
                       choices=['prompt_tuning', 'traditional', 'lora'],
                       required=True)
    parser.add_argument('--max_article_length', type=int, default=512)
    parser.add_argument('--max_summary_length', type=int, default=128)
    
    # Training strategy specific arguments
    parser.add_argument('--num_virtual_tokens', type=int, default=20)
    parser.add_argument('--num_layers_to_finetune', type=int, default=2)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # Logging arguments
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='gpt2-summarization')
    parser.add_argument('--run_name', type=str, default='experiment')
    
    return parser.parse_args()

def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()