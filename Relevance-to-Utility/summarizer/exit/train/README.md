# Training EXIT Model

This guide walks through the complete process of preparing data and training the EXIT model.

## 1. Dataset Preparation

EXIT is trained on the HotpotQA dataset, using sentences labeled as supporting facts as positive examples and carefully selected negative examples.

### 1.1 Download HotpotQA Dataset
```bash
# Create data directory
mkdir -p data/hotpotqa
cd data/hotpotqa

# Download training set
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
```

### 1.2 Process Training Data
```bash
# Process the dataset
python train/preprocess_hotpotqa.py \
    --dataset_path data/hotpotqa/hotpot_train_v1.1.json \
    --save_dir data/processed \
    --test_size 1000 \
    --positive_ratio 0.5
```

This creates:
- `data/processed/train_dataset/`: Training dataset
- `data/processed/test_dataset/`: Test dataset (last 1000 examples)

## 2. Model Training

### 2.1 Environment Setup
```bash
# Create conda environment
conda create -n exit python=3.8
conda activate exit

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies for experiment tracking
pip install wandb
```

### 2.2 Training Configuration

The training script supports various configurations. Key parameters include:

```bash
python train/train.py \
    --model_id "google/gemma-2b-it" \
    --train_dataset "data/processed/train_dataset" \
    --test_dataset "data/processed/test_dataset" \
    --output_dir "outputs/exit_model" \
    --wandb_project "exit" \      # Optional: for experiment tracking
    --experiment_name "exit_gemma_2b"
```

#### Key Training Parameters:
- `--lora_r`: LoRA rank (default: 64)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--per_device_train_batch_size`: Batch size per GPU (default: 8)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 8)
- `--num_train_epochs`: Number of training epochs (default: 1)

### 2.3 Training Process

1. **Initial Setup**:
```bash
# Create output directory
mkdir -p outputs/exit_model
```

2. **Basic Training**:
```bash
python train/train.py \
    --model_id "google/gemma-2b-it" \
    --train_dataset "data/processed/train_dataset" \
    --test_dataset "data/processed/test_dataset" \
    --output_dir "outputs/exit_model"
```

3. **Training with Experiment Tracking**:
```bash
# First, login to wandb (one-time setup)
wandb login

# Start training with tracking
python train/train.py \
    --model_id "google/gemma-2b-it" \
    --train_dataset "data/processed/train_dataset" \
    --test_dataset "data/processed/test_dataset" \
    --output_dir "outputs/exit_model" \
    --wandb_project "exit" \
    --experiment_name "exit_gemma_2b"
```

4. **Resume Training from Checkpoint**:
```bash
python train/train.py \
    --model_id "google/gemma-2b-it" \
    --train_dataset "data/processed/train_dataset" \
    --test_dataset "data/processed/test_dataset" \
    --output_dir "outputs/exit_model" \
    --resume_from_checkpoint "outputs/exit_model/checkpoint-1000"
```

### 2.4 Training Output

The training process will create:
- Model checkpoints at regular intervals
- Evaluation metrics
- Final model weights
- Training logs

Training progress can be monitored via:
- Command line outputs
- Tensorboard (if enabled)
- Weights & Biases dashboard (if configured)

## 3. Model Evaluation

After training, you can evaluate the model:

```bash
python scripts/evaluate.py \
    --base_model "google/gemma-2b-it" \
    --checkpoint "outputs/exit_model/checkpoint-1000" \
    --test_dataset "data/processed/test_dataset" \
    --output_dir "outputs/evaluation" \
    --threshold 0.5 \
    --batch_size 16
```
