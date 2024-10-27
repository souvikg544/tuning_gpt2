import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import json
import pandas as pd
from typing import Dict, List, Union

class SummarizationDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: GPT2Tokenizer,
        max_article_length: int = 512,
        max_summary_length: int = 128,
        file_type: str = 'json'
    ):
        """
        Initialize the summarization dataset.
        
        Args:
            data_path: Path to the data file (json or csv)
            tokenizer: Tokenizer to use for encoding
            max_article_length: Maximum length for articles
            max_summary_length: Maximum length for summaries
            file_type: Type of data file ('json' or 'csv')
        """
        self.tokenizer = tokenizer
        self.max_article_length = max_article_length
        self.max_summary_length = max_summary_length
        
        df = pd.read_csv(data_path)
        self.data = df.to_dict('records')
        
        # # Add special tokens for summarization
        # special_tokens = {
        #     'sep_token': '<|sep|>',
        #     'pad_token': '<|pad|>',
        #     'bos_token': '<|startoftext|>',
        #     'eos_token': '<|endoftext|>'
        # }
        
        # # Add special tokens to tokenizer
        # self.tokenizer.add_special_tokens({
        #     'additional_special_tokens': list(special_tokens.values())
        # })
        
        # Get special token ids
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids('<|sep|>')
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids('<|pad|>')
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids('<|startoftext|>')
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')

        
    def __len__(self) -> int:
        return len(self.data)
    
    def prepare_input(self, article: str, summary: str) -> Dict[str, torch.Tensor]:
        """
        Prepare input by tokenizing and formatting article and summary.
        """
        # Format: <|startoftext|> article <|sep|> summary <|endoftext|>
        # Tokenize article and summary separately
        article_tokens = self.tokenizer.encode(
            article,
            max_length=self.max_article_length,
            truncation=True,
            add_special_tokens=False
        )
        
        summary_tokens = self.tokenizer.encode(
            summary,
            max_length=self.max_summary_length,
            truncation=True,
            add_special_tokens=False
        )
        
        # Combine tokens with special tokens
        input_ids = (
            [self.bos_token_id] +  # Start token
            article_tokens +
            [self.sep_token_id] +  # Separator
            summary_tokens +
            [self.eos_token_id]    # End token
        )
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Create labels for training
        # -100 is the ignore index for CrossEntropyLoss
        labels = (
            [-100] * (len(article_tokens) + 2) +  # +2 for bos and sep tokens
            summary_tokens +
            [self.eos_token_id]
        )
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Get article and summary from data
        # Adjust these field names based on your data structure
        article = item.get('article', item.get('body', ''))
        summary = item.get('highlights', item.get('highlights', ''))
        
        # Prepare the input
        encoded = self.prepare_input(article, summary)
        
        # Convert to tensors
        return {
            'input_ids': torch.tensor(encoded['input_ids']),
            'attention_mask': torch.tensor(encoded['attention_mask']),
            'labels': torch.tensor(encoded['labels'])
        }

def create_dataloader(
    dataset: SummarizationDataset,
    batch_size: int = 8,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a dataloader with padding collation.
    """
    def collate_fn(batch):
        # Find max length in batch
        max_length = max(len(item['input_ids']) for item in batch)
        
        # Initialize padded batch
        padded_batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        # Pad each item to max_length
        for item in batch:
            for key in padded_batch:
                padding_length = max_length - len(item[key])
                if key == 'labels':
                    padding_value = -100
                else:
                    padding_value = dataset.pad_token_id if key == 'input_ids' else 0
                    
                padded_item = torch.cat([
                    item[key],
                    torch.ones(padding_length, dtype=torch.long) * padding_value
                ])
                padded_batch[key].append(padded_item)
        
        # Stack all items in batch
        return {
            key: torch.stack(value)
            for key, value in padded_batch.items()
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

# Example usage
def demonstrate_usage():
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Create dataset
    dataset = SummarizationDataset(
        '/ssd_scratch/cvit/souvik/cnn_dailymail/train.csv',
        tokenizer,
        max_article_length=128,
        max_summary_length=32
    )
    
    # Create dataloader
    dataloader = create_dataloader(dataset, batch_size=2)
    
    # Get a batch
    batch = next(iter(dataloader))
    
    # Print example
    print("Input text:")
    print(tokenizer.decode(batch['input_ids'][0]))
    print("\nLabels (showing only summary part):")
    print(tokenizer.decode([
        token for token in batch['labels'][0] if token != -100
    ]))

if __name__ == "__main__":
    demonstrate_usage()