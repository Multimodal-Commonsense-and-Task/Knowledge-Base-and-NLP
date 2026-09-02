# EMBGuard

EMBGuard is the first MLLM-based safety guardrail framework for embodied agents operating in real-world environments. It identifies action-conditioned physical hazards from (visual observation, action) pairs, provides risk explanations, and enables safe planning through decoupled risk reasoning.

## 📦 Datasets on Hugging Face

All EMBGuard datasets and models are available on Hugging Face:

🔗 **https://huggingface.co/EMBGuard**

### Available Datasets

- **EMBHazard**: Training dataset for safety guardrail models
  - 15.1K samples for training

- **EMBGuardTest**: Test set for evaluation
  - Contains 4 scenario types: Causal Risky, Decoupled Benign, Selective Risky, Absent Benign
  - Available as Hugging Face splits: `causal_risky`, `decoupled_benign`, `selective_risky`, `absent_benign`

- **Heldout Set**: Additional evaluation dataset
  - Available as Hugging Face splits: `causal_risky`, `decoupled_benign`, `selective_risky`, `absent_benign`
  - Legacy `safe` and `unsafe` splits are also retained for compatibility

### Available Models

- `EMBGuard/EMBGuard-2B` - 2B parameter model
- `EMBGuard/EMBGuard-4B` - 4B parameter model
- `EMBGuard/EMBGuard-8B` - 8B parameter model
- Various LoRA fine-tuned variants

## Terminology

EMBGuardTest is organized around four evaluation conditions:

- **Causal Risky**: A hazard is present and causally relevant to the action, so the scene should be judged unsafe.
- **Selective Risky**: Multiple hazards or hazard candidates are present, and the model must identify the action-relevant risky condition.
- **Decoupled Benign**: A hazard-like element is present but decoupled from the action, so the scene should be judged safe.
- **Absent Benign**: The hazard is absent, so the scene should be judged safe.

The Hugging Face split names are `causal_risky`, `selective_risky`, `decoupled_benign`, and `absent_benign`.

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd EMBGuard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .  # Install in editable mode
```

3. Configure API keys in `conf.d/config.yaml`:
```yaml
openai:
  key: your-openai-api-key

openrouter:
  key: your-openrouter-api-key

anthropic:
  key: your-anthropic-api-key

gemini:
  key: your-gemini-api-key
```

### Quick Evaluation

Run evaluation on test set and heldout set:
```bash
bash scripts/evaluation/run_all_evaluations.sh
```

## 📁 Project Structure

The project is organized around the `scripts/` directory, which contains all executable workflows:

```
EMBGuard/
├── scripts/                          # All executable scripts
│   ├── train/                        # Model training scripts
│   │   ├── 1_download_EMBHazard_dataset.sh
│   │   ├── 2_construct_train_data.sh
│   │   ├── train_qwen3vl_lora.sh    # LoRA fine-tuning
│   │   ├── train_qwen3vl_full.sh    # Full fine-tuning
│   │   ├── merge_adapter.sh         # Merge LoRA adapters
│   │   └── upload_merged_models.sh  # Upload to Hugging Face
│   │
│   ├── evaluation/                   # Evaluation scripts
│   │   ├── run_all_evaluations.sh   # Main evaluation script
│   │   ├── aggregate_results.sh    # Aggregate multiple runs
│   │   ├── run_calculate_correlation.sh  # Calculate correlations
│   │   └── benchmark_vllm_latency.sh    # Latency benchmarking
│   │
│   ├── visualization/                # Visualization scripts
│   │   ├── plot_correlation.sh      # Plot correlation results
│   │   └── plot_type_trend.sh       # Plot scenario_type-specific trends
│   │
│   ├── dataset_generation/           # Dataset generation pipeline
│   │   ├── 1_make_scenarios.sh      # Step 1: Scenario generation
│   │   ├── 2_make_graphs.sh         # Step 2: Graph generation
│   │   ├── 3_1_1_scene_augmentation.sh
│   │   ├── 3_1_2_hazard_removal.sh
│   │   ├── 3_2_hazard_augmentation.sh
│   │   ├── 3_3_merge_graphs.sh
│   │   ├── 4_graph_to_text.sh      # Step 4: Text generation
│   │   ├── 5_1_text_to_image_w_batch.sh  # Step 5: Image generation
│   │   ├── 5_2_check_batch.sh
│   │   ├── 5_3_download_image.sh
│   │   ├── 6_action_augmentation.sh # Step 6: Action augmentation
│   │   └── 7_make_dataset.sh        # Step 7: Final dataset creation
│   │
│   ├── upload_to_huggingface.sh     # Upload datasets/models to HF
│   └── run_vllm.sh                  # Start vLLM server
│
├── src/                              # Source code
│   ├── guardrail/                    # Guardrail model implementation
│   ├── evals/                        # Evaluation modules
│   ├── visualization/                # Visualization code
│   └── evaluate.py                  # CLI entry point
│
├── conf.d/                           # Configuration files
│   └── config.yaml                   # Main configuration
│
├── data/                             # Local data storage
│   ├── test_set/                     # Test set CSV files and images
│   └── heldout_set/                  # Heldout set CSV files and images
│
├── outputs/                          # Inference results (JSONL)
│   ├── EMBGuardTest/
│   └── heldout_set/
│
├── results/                          # Evaluation results (JSON)
│   ├── EMBGuardTest/
│   ├── heldout_set/
│   └── figures/                     # Generated plots
│
└── LlamaFactory/                     # LlamaFactory for training
```

## 📚 Scripts Guide

### 1. Training Scripts (`scripts/train/`)

#### Download Dataset
```bash
bash scripts/train/1_download_EMBHazard_dataset.sh
```
Downloads the EMBHazard dataset from Hugging Face.

#### Construct Training Data
```bash
bash scripts/train/2_construct_train_data.sh
```
Prepares training data in the format required by LlamaFactory.

#### Train with LoRA
```bash
bash scripts/train/train_qwen3vl_lora.sh
```
Fine-tunes Qwen3-VL models using LoRA. Configuration is in `train_qwen3vl_lora.yaml`.

**Configuration** (`train_qwen3vl_lora.yaml`):
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Dataset: `embhazard_wo_filter_v1` (from Hugging Face)
- Training method: LoRA
- Cutoff length: 2048

#### Train Full Fine-tuning
```bash
bash scripts/train/train_qwen3vl_full.sh
```
Full fine-tuning (not LoRA). Configuration in `qwen3vl_full_sft.yaml`.

#### Merge LoRA Adapters
```bash
bash scripts/train/merge_adapter.sh
```
Merges LoRA adapters into the base model for deployment.

#### Upload Models
```bash
bash scripts/train/upload_merged_models.sh
```
Uploads trained models to Hugging Face Hub.

#### Migrate Hugging Face Datasets
```bash
python -m src.hf_utils.migrate_hf_datasets \
  --org EMBGuard \
  --datasets EMBGuardTest_v2 heldout_set EMBHazard \
  --dry-run
```
Rebuilds existing Hub datasets from Hub sources without requiring local CSV/image files. The default `v3` schema normalizes public columns to `id`, `image`, `image_path`, `risk`, `type`, `scenario_type`, `risk_type`, `risk_subtype`, `situation`, `hazard`, `action`, `room`, `multi_scenario`, and `pair_item_id`. Legacy display columns such as `Type Label`, `Subtype Label`, `Category`, `Subcategory`, `Risk Type`, `Mitigate Action`, and `URL` are replaced by the normalized names. For heldout_set, the legacy `safe` and `unsafe` splits are retained by default alongside `causal_risky`, `decoupled_benign`, `selective_risky`, and `absent_benign`. Use `--schema legacy-labels` only if you need the previous legacy column layout.

### 2. Evaluation Scripts (`scripts/evaluation/`)

#### Run All Evaluations
```bash
bash scripts/evaluation/run_all_evaluations.sh
```
Main evaluation script that:
- Runs test set evaluation (Causal Risky, Decoupled Benign, Selective Risky, Absent Benign)
- Runs heldout set evaluation (safe, unsafe)
- Automatically evaluates results using LLM-as-a-judge
- Supports multiple runs with `Run_n/` folder structure

**Configuration** (edit in script):
```bash
# Provider-Model pairs to evaluate
PROVIDER_MODEL_PAIRS=(
    "openai:gpt-4o"
    "vllm:Qwen/Qwen3-VL-4B-Instruct:8000"
)

# Test set configuration
TEST_SET="all"  # Options: "all", "causal_risky", "selective_risky", "decoupled_benign", "absent_benign", or comma-separated
TEST_SET_NUM_WORKERS="16"

# Heldout set configuration
HELDOUT_DATASET="all"  # Options: "all", "safe", "unsafe"
HELDOUT_NUM_WORKERS="32"

# Evaluation options
USE_FEW_SHOT="false"
USE_THINKING="false"

# Number of runs
NUM_RUNS=5  # Results saved in Run_1/, Run_2/, ... folders
```

#### Aggregate Results
```bash
bash scripts/evaluation/aggregate_results.sh
```
Aggregates results across multiple runs:
- Calculates mean and 95% confidence intervals
- Generates CSV files:
  - `aggregated_results_overall.csv`
  - `aggregated_results_overall_percentage.csv`
  - `aggregated_results_by_scenario_type.csv`
- Includes conditional accuracy metrics

#### Calculate Correlation
```bash
bash scripts/evaluation/run_calculate_correlation.sh
```
Calculates correlation between test set and heldout set scores:
- Supports multiple metrics: `potential_risk`, `conditional_risk_type`, `conditional_hazard`
- Calculates both Pearson and Spearman correlations
- Generates CSV files and plots

### 3. Visualization Scripts (`scripts/visualization/`)

#### Plot Correlation
```bash
bash scripts/visualization/plot_correlation.sh
```
Creates correlation plots showing the relationship between test set and heldout set scores:
- Plots all three metrics in a single figure
- Different colors for each metric
- Saves to `results/figures/`

#### Plot Scenario Type Trends
```bash
bash scripts/visualization/plot_type_trend.sh
```
Visualizes model performance across scenario types (Causal Risky, Selective Risky, Decoupled Benign, Absent Benign):
- Shows potential risk accuracy trends
- Background colors: red for risky (Causal Risky/Selective Risky), green for benign (Decoupled Benign/Absent Benign)
- Customizable model selection and styling
- Saves to `results/figures/`

### 4. Dataset Generation Scripts (`scripts/dataset_generation/`)

Complete pipeline for generating safety datasets. See `scripts/dataset_generation/README.md` for detailed documentation.

**Workflow**:
1. **Scenario Generation** (`1_make_scenarios.sh`) - Taxonomy → Scenarios
2. **Graph Generation** (`2_make_graphs.sh`) - Scenarios → Graphs
3. **Graph Post-processing**:
   - Scene augmentation (`3_1_1_scene_augmentation.sh`)
   - Hazard removal (`3_1_2_hazard_removal.sh`)
   - Hazard augmentation (`3_2_hazard_augmentation.sh`)
   - Merge graphs (`3_3_merge_graphs.sh`)
4. **Text Generation** (`4_graph_to_text.sh`) - Graph → Text
5. **Image Generation** (`5_1_text_to_image_w_batch.sh`, `5_2_check_batch.sh`, `5_3_download_image.sh`)
6. **Action Augmentation** (`6_action_augmentation.sh`) - Safe action variants
7. **Dataset Creation** (`7_make_dataset.sh`) - Final CSV dataset

## 🔧 Configuration

### Config File (`conf.d/config.yaml`)

Supports both local CSV files and Hugging Face datasets:

```yaml
common:
  data_dir: data/test_set
  test_set:
    # Option 1: Local CSV files
    causal_risky: data/test_set/test_dataset_HR.csv
    selective_risky: data/test_set/test_dataset_MHR.csv
    decoupled_benign: data/test_set/test_dataset_HNR.csv
    absent_benign: data/test_set/test_dataset_NHR.csv
    
    # Option 2: Hugging Face dataset (recommended)
    causal_risky: EMBGuard/EMBGuardTest_v2
    selective_risky: EMBGuard/EMBGuardTest_v2
    decoupled_benign: EMBGuard/EMBGuardTest_v2
    absent_benign: EMBGuard/EMBGuardTest_v2
    
  heldout_set:
    safe: EMBGuard/heldout_set
    unsafe: EMBGuard/heldout_set
    
  use_thinking: false  # Enable thinking mode
```

The system automatically detects Hugging Face datasets (paths containing "/" that don't exist as files). Use the full split names for new workflows: `causal_risky`, `selective_risky`, `decoupled_benign`, and `absent_benign`. Legacy aliases such as `HR`, `hr`, `HNR`, `MHR`, and `NHR` remain supported for older configs and results.

## 📊 Evaluation Types

### Test Sets (EMBGuardTest)

- **Causal Risky** (`HR` split): Scenes with hazards that should be detected as risky
- **Decoupled Benign** (`HNR` split): Scenes with hazards but should be safe (decoupled)
- **Selective Risky** (`MHR` split): Multiple hazards, should be risky
- **Absent Benign** (`NHR` split): No hazards, should be safe

### Heldout Sets

- **Causal Risky** (`causal_risky` split): 111 heldout unsafe scenes
- **Selective Risky** (`selective_risky` split): 250 heldout unsafe scenes
- **Decoupled Benign** (`decoupled_benign` split): 112 heldout safe scenes
- **Absent Benign** (`absent_benign` split): 92 heldout safe scenes

The legacy `safe` and `unsafe` splits are still available for compatibility.

### Evaluation Metrics

1. **Potential Risk Accuracy**: Binary classification (safe/unsafe)
2. **Risk Type Accuracy**: Categorical classification of risk types (for unsafe cases)
3. **Hazard Accuracy**: LLM-as-a-judge evaluation comparing predicted vs. ground truth hazards
4. **Conditional Accuracies**: Risk type and hazard accuracy when potential risk is correctly identified

## 🎯 Usage Examples

### Example 1: Evaluate a Single Model

```bash
# Edit scripts/evaluation/run_all_evaluations.sh
PROVIDER_MODEL_PAIRS=(
    "openai:gpt-4o"
)

# Run evaluation
bash scripts/evaluation/run_all_evaluations.sh
```

### Example 2: Train and Evaluate Custom Model

```bash
# 1. Download dataset
bash scripts/train/1_download_EMBHazard_dataset.sh

# 2. Prepare training data
bash scripts/train/2_construct_train_data.sh

# 3. Train with LoRA
bash scripts/train/train_qwen3vl_lora.sh

# 4. Merge adapters
bash scripts/train/merge_adapter.sh

# 5. Start vLLM server with your model
bash scripts/run_vllm.sh

# 6. Evaluate
bash scripts/evaluation/run_all_evaluations.sh
```

### Example 3: Multiple Runs and Aggregation

```bash
# 1. Run evaluation 5 times
# Edit NUM_RUNS=5 in run_all_evaluations.sh
bash scripts/evaluation/run_all_evaluations.sh

# 2. Aggregate results
bash scripts/evaluation/aggregate_results.sh

# 3. Calculate correlations
bash scripts/evaluation/run_calculate_correlation.sh

# 4. Visualize
bash scripts/visualization/plot_correlation.sh
bash scripts/visualization/plot_type_trend.sh
```

## 🤖 Supported Models

### OpenAI
- `gpt-4o`, `gpt-4o-mini`, `gpt-5.1`

### OpenRouter
- Any model available on OpenRouter
- Examples: `qwen/qwen3-vl-8b-instruct`, `gemini-2.5-pro`

### vLLM (Local)
- Any model compatible with vLLM
- Examples: `Qwen/Qwen3-VL-4B-Instruct`, `EMBGuard/EMBGuard-4B`
- Start server: `bash scripts/run_vllm.sh`

### Claude (Anthropic)
- `claude-3-5-sonnet-20241022`

### Gemini (Google)
- `gemini-2.5-flash`, `gemini-2.5-pro`

## 📤 Output Structure

### Inference Results (`outputs/`)
```
outputs/
├── EMBGuardTest/
│   └── Run_1/                    # Run number
│       └── {provider}_{model}/
│           └── {model}_{type}_{condition}_results.jsonl
└── heldout_set/
    └── Run_1/
        └── {provider}_{model}/
            └── {model}_{dataset}_{condition}_results.jsonl
```

### Evaluation Results (`results/`)
```
results/
├── EMBGuardTest/
│   └── Run_1/
│       └── {provider}_{model}/
│           └── {model}_{type}_{condition}_evaluation.json
│   └── aggregated_results_overall.csv
│   └── aggregated_results_by_scenario_type.csv
└── heldout_set/
    └── Run_1/
        └── {provider}_{model}/
            └── {model}_{dataset}_{condition}_evaluation.json
└── figures/
    ├── correlation_combined_*.png
    └── scenario_type_trend_potential_risk.png
```

## 🔍 Advanced Features

### Thinking Mode
Enable step-by-step reasoning:
```bash
USE_THINKING="true"  # In run_all_evaluations.sh
```

### Few-Shot Examples
Enable few-shot examples in prompts:
```bash
USE_FEW_SHOT="true"  # In run_all_evaluations.sh
```

### Parallel Processing
Configure workers for parallel inference:
- Test set: `TEST_SET_NUM_WORKERS="16"`
- Heldout set: `HELDOUT_NUM_WORKERS="32"`
- Results evaluation: `RESULTS_NUM_WORKERS="16"`

## 🐛 Troubleshooting

### vLLM Connection Issues
- Ensure vLLM server is running: `bash scripts/run_vllm.sh`
- Check port number matches in `PROVIDER_MODEL_PAIRS`
- Verify `base_url` in `config.yaml` matches your vLLM server

### Hugging Face Dataset Access
- Set `HF_TOKEN` environment variable or in scripts
- For private datasets, ensure you have access permissions

### Image Loading Errors
- For local CSV: Ensure image paths in CSV are relative to `data_dir` in config
- For Hugging Face: Images are automatically loaded from the dataset

### API Key Errors
- Verify API keys are correctly set in `conf.d/config.yaml`
- Check API key permissions and quotas

## 📝 Citation

If you use EMBGuard in your research, please cite:

```bibtex
@article{choi2026embguard,
  title={EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents},
  author={Choi, Dongwook and Kwon, Taeyoon and Jeong, Bogyung and Kim, Minju and Hwang, Yeonjun and Kim, Hyojun and Kim, Byungchul and Jang, Young Kyun and Yeo, Jinyoung},
  journal={arXiv preprint arXiv:2605.30924},
  year={2026}
}
```

## 📄 License

The code in this repository is released under the [MIT License](LICENSE).

Datasets, model weights, and third-party components may be subject to their own licenses or the licenses of their underlying sources. Please check the corresponding Hugging Face dataset/model cards and third-party repositories before redistribution or commercial use.
