Usage 
Training
```
CUDA_VISIBLE_DEVICES=0 python train.py --training_strategy prompt_tuning --train_data_path /ssd_scratch/cvit/souvik/cnn_dailymail/train_small.csv --val_data_path /ssd_scratch/cvit/souvik/cnn_dailymail/val_small.csv --output_dir /ssd_scratch/cvit/souvik/outputs/prompt --wandb --run_name prompt
```


CUDA_VISIBLE_DEVICES=1 python train.py --training_strategy lora --train_data_path /ssd_scratch/cvit/souvik/cnn_dailymail/train_small.csv --val_data_path /ssd_scratch/cvit/souvik/cnn_dailymail/val_small.csv --output_dir /ssd_scratch/cvit/souvik/outputs/lora --wandb --run_name lora


CUDA_VISIBLE_DEVICES=2 python train.py --training_strategy traditional --train_data_path /ssd_scratch/cvit/souvik/cnn_dailymail/train_small.csv --val_data_path /ssd_scratch/cvit/souvik/cnn_dailymail/val_small.csv --output_dir /ssd_scratch/cvit/souvik/outputs/finetune --wandb --run_name finetune