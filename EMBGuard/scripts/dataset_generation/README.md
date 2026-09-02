# Safety Dataset Generation Pipeline Scripts

This directory contains pipeline scripts for safety dataset generation.

## 📋 Overall Workflow

```
1. Scenario Generation → 2. Graph Generation → 3. Graph Post-processing → 4. Text Generation → 5. Image Generation → 6. Action Augmentation → 7. Dataset Generation
```

---

## 🚀 Step-by-Step Script Guide

### 1. Scenario Generation
**File**: `1_make_scenarios.sh`
**Purpose**: Taxonomy → Scenarios conversion

**Values to modify**:
```bash
ITERATE_NAME="test"                       # Result folder name
SHOTS_FILE="shots_w_mechanism_sample.json"  # Input shots file
NUM_SCENARIOS=0                           # Number of scenarios to generate (0=use shots only)
TAXONOMY_MODEL="gpt-5.2"                 # Scenario generation model
TAXONOMY_MODEL_TYPE="gpt"                # Model type
```

**Result**: `scenarios.json`

---

### 2. Graph Generation
**File**: `2_make_graphs.sh`
**Purpose**: Scenarios → Graphs → Normalization

**Values to modify**:
```bash
ITERATE_NAME="test"                       # Same as step 1
SCENARIOS_FILE="scenarios.json"          # Step 1 result file
GRAPH_MODEL="gpt-5.2"                    # Graph generation model
GRAPH_MODEL_TYPE="gpt"                   # Model type
NORM_MODEL="gpt-5.2"                     # Normalization model
NORM_MODEL_TYPE="gpt"                    # Model type
```

**Result**: `graphs_normalized.json`

---

### 3. Graph Post-processing (3 sub-steps)

#### 3.1.1 Scene Augmentation
**File**: `3_1_1_scene_augmentation.sh`
**Purpose**: Place additional objects in scene

**Values to modify**:
```bash
ITERATE_NAME="test"                       # Same as previous step
INPUT_FILE="graphs_normalized.json"      # Step 2 result file
SCENE_AUG_MODEL="gpt-5.2"               # Scene augmentation model
SCENE_AUG_MODEL_TYPE="gpt"              # Model type
```

**Result**: `graphs_scene_augmented.json`

#### 3.1.2 Hazard Removal  
**File**: `3_1_2_hazard_removal.sh`
**Purpose**: Remove hazard elements

**Values to modify**:
```bash
ITERATE_NAME="test"                       # Same as previous step
INPUT_FILE="graphs_normalized.json"      # Step 2 result file
HAZARD_REMOVAL_MODEL="gpt-5.2"          # Hazard removal model
HAZARD_REMOVAL_MODEL_TYPE="gpt"         # Model type
```

**Result**: `graphs_hazard_removed.json`

#### 3.2 Hazard Augmentation
**File**: `3_2_hazard_augmentation.sh`
**Purpose**: Generate multiple hazard scenarios

**Values to modify**:
```bash
ITERATE_NAME="test"                       # Same as previous step
INPUT_FILE="graphs_normalized.json"      # Step 2 result file
HAZARD_AUG_MODEL="gpt-5.2"              # Hazard augmentation model
HAZARD_AUG_MODEL_TYPE="gpt"             # Model type
```

**Result**: `graphs_hazard_augmented.json`

#### 3.3 Graph Merging
**File**: `3_3_merge_graphs.sh`
**Purpose**: Merge all graph variations into one

**Values to modify**:
```bash
ITERATE_NAME="test"                       # Same as previous step
```

**Result**: `graphs_final.json` (includes all merge_source)

---

### 3. Graph Merging
**File**: `3_3_merge_graphs.sh`
**Purpose**: Merge all graph variations into one

**Values to modify**:
```bash
ITERATE_NAME="heldout"                    # Same as previous step
```

**Result**: `graphs_final.json` (includes all merge_source)

---

### 4. Text Generation
**File**: `4_graph_to_text.sh`
**Purpose**: Graph → Text conversion

**Values to modify**:
```bash
ITERATE_NAME="test"                       # Same as previous step
INPUT_FILE="graphs_final.json"           # Step 3.3 result file
TEXT_MODEL="gpt-5.2"                     # Text generation model
TEXT_MODEL_TYPE="gpt"                    # Model type
```

**Result**: `texts_generated.json`

---

### 5. Image Generation (Batch)

#### 5.1 Create Batch Job
**File**: `5_1_text_to_image_w_batch.sh`
**Values to modify**:
```bash
ITERATE_NAME="test"                       # Same as previous step
INPUT_FILE="texts_generated.json"        # Step 4 result file
IMAGE_MODEL="gemini-3-pro-image-preview" # Image generation model
IMAGE_MODEL_TYPE="openrouter_image"      # Model type
```

**Result**: Batch job created and submitted to Google AI

#### 5.2 Check Batch Status
**File**: `5_2_check_batch.sh`
**Values to modify**:
```bash
BATCH_JOB_NAME="batches/YOUR_JOB_ID"     # Job ID created in 5.1
BATCH_RUN_TIMESTAMP="1234567890"        # Timestamp created in 5.1
```

#### 5.3 Download Images
**File**: `5_3_download_image.sh`
**Values to modify**:
```bash
BATCH_JOB_NAME="batches/YOUR_JOB_ID"     # Same as 5.2
BATCH_RUN_TIMESTAMP="1234567890"        # Same as 5.2
MERGE_SOURCE="scene_augmented"          # merge_source to download
```

**Result**: `texts_with_images_{merge_source}.json`

**💡 Tip**: Repeat 5.3 for each merge_source:
- `MERGE_SOURCE="scene_augmented"`
- `MERGE_SOURCE="hazard_removed"` 
- `MERGE_SOURCE="hazard_augmented"`

---

### 6. Action Augmentation
**File**: `6_action_augmentation.sh`
**Purpose**: Change to safe actions

**Values to modify**:
```bash
ITERATE_NAME="test"                      # Same as previous step
INPUT_FILE="texts_with_images_scene_augmented.json"  # Step 5.3 result file
ACTION_AUG_MODEL="gpt-5.2"              # Action augmentation model
ACTION_AUG_MODEL_TYPE="gpt"             # Model type
OUTPUT_FILE="texts_with_images_scene_augmented_safe.json"  # Output filename
```

**Result**: `texts_with_images_scene_augmented_safe.json`

---

### 7. Dataset Generation
**File**: `7_make_dataset.sh`
**Purpose**: Integrate all JSON files into CSV dataset

**Values to modify**:
```bash
DEFAULT_ITERATE_NAME="test"              # Same as previous step
DEFAULT_INPUT_FILES=(                    # Files to process
  "texts_with_images_scene_augmented.json"
  "texts_with_images_hazard_removed.json"
  "texts_with_images_scene_augmented_safe.json"
  "texts_with_images_hazard_augmented.json"
)
DEFAULT_OUTPUT_CSV="complete/dataset_all_with_images.csv"  # Output file
```

**Result**: `complete/dataset_all_with_images.csv`

**Automatic processing**:
- Auto split `hazard_augmented` file (dual → individual scenarios)
- Auto classify Subtype:
  - Causal Risky (`HR`): hazard_augmented, scene_augmented
  - Absent Benign (`NHR`): hazard_removed
  - Decoupled Benign (`HNR`): scene_augmented_safe

---

## 🔧 Common Settings

### Environment Variables
```bash
export OPENAI_API_KEY="your_openai_key"
export GOOGLE_API_KEY="your_google_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### Recommended Models
- **Text Generation**: `gpt-4o`, `gpt-4o-mini`
- **Image Generation**: `gemini-3-pro-image-preview`
- **Action Augmentation**: `gpt-5.2`

---

## 📁 Result Structure

```
results/{ITERATE_NAME}/
├── graphs_normalized.json              # Step 1
├── graphs_scene_augmented.json         # Step 2
├── graphs_hazard_removed.json          # Step 2
├── graphs_hazard_augmented.json        # Step 2
├── graphs_final.json                   # Step 3
├── texts_generated.json                # Step 4
├── texts_with_images_*.json             # Step 5
├── texts_with_images_*_safe.json       # Step 6
├── images/downloaded/                   # Step 5 images
└── complete/dataset_all_with_images.csv # Step 7 final dataset
```

---

## 🚨 Notes

1. **Sequential execution**: Each step depends on the results of the previous step
2. **ITERATE_NAME consistency**: Use the same value in all scripts
3. **Batch jobs**: Step 5 may take a long time (several hours)
4. **API costs**: Consider costs when generating large amounts
5. **merge_source processing**: Steps 5.3 and 6 are executed for each merge_source

---

## 🔄 Example Execution Order

```bash
# 1. Scenario Generation
bash scripts/1_make_scenarios.sh

# 2. Graph Generation  
bash scripts/2_make_graphs.sh

# 3. Graph Post-processing
bash scripts/3_1_1_scene_augmentation.sh
bash scripts/3_1_2_hazard_removal.sh
bash scripts/3_2_hazard_augmentation.sh
bash scripts/3_3_merge_graphs.sh

# 4. Text Generation
bash scripts/4_graph_to_text.sh

# 5. Image Generation (Batch)
bash scripts/5_1_text_to_image_w_batch.sh
# (Wait for batch completion)
bash scripts/5_2_check_batch.sh
bash scripts/5_3_download_image.sh  # scene_augmented
# After changing MERGE_SOURCE, also run for hazard_removed, hazard_augmented

# 6. Action Augmentation
bash scripts/6_action_augmentation.sh

# 7. Dataset Generation
bash scripts/7_make_dataset.sh
```

Final output: `results/{ITERATE_NAME}/complete/dataset_all_with_images.csv`
