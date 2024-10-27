### GPT 2 cries in the corner (LOL)
This repository performs the task of summarization on GPT2 model. The code is inspired from an assignment in Advanced NLP course taken Professor Manish Srivastava in IIIT Hyderabad.
The different procedures are as follows - 
- Prompt Tuning
```
CUDA_VISIBLE_DEVICES=0 python train.py --training_strategy prompt_tuning --train_data_path /ssd_scratch/cvit/souvik/cnn_dailymail/train_small.csv --val_data_path /ssd_scratch/cvit/souvik/cnn_dailymail/val_small.csv --output_dir /ssd_scratch/cvit/souvik/outputs/prompt --wandb --run_name prompt
```

- LORA Adaptation
```
CUDA_VISIBLE_DEVICES=1 python train.py --training_strategy lora --train_data_path /ssd_scratch/cvit/souvik/cnn_dailymail/train_small.csv --val_data_path /ssd_scratch/cvit/souvik/cnn_dailymail/val_small.csv --output_dir /ssd_scratch/cvit/souvik/outputs/lora --wandb --run_name lora
```
- Traditional Finetuning
```
CUDA_VISIBLE_DEVICES=2 python train.py --training_strategy traditional --train_data_path /ssd_scratch/cvit/souvik/cnn_dailymail/train_small.csv --val_data_path /ssd_scratch/cvit/souvik/cnn_dailymail/val_small.csv --output_dir /ssd_scratch/cvit/souvik/outputs/finetune --wandb --run_name finetune
```

The model weights can be found [here](https://drive.google.com/drive/folders/1xnvZSqZApEaaECMQnGXoQw9EGLQRZ1iJ?usp=sharing)
