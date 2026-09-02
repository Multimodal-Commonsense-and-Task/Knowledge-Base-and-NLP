# PAC-BENCH: Evaluating Multi-Agent Collaboration under Privacy Constraints

<p align="center">
	<img src="_asset/logo.png" alt="PAC-BENCH Logo" width="90px">
	<br>
	<b>A benchmark for evaluating multi-agent collaboration under privacy constraints</b>
</p>

<p align="center">
<a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11"></a>
<a href="https://huggingface.co/datasets/PAC-Bench/PAC-Bench"><img src="https://img.shields.io/badge/HuggingFace-Dataset-ffcc4d.svg" alt="Hugging Face Dataset"></a>
<a href="https://arxiv.org/abs/0000.00000"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="arXiv Paper"></a>
</p>

![Overview](_asset/overview.jpg)

PAC-BENCH is a benchmark pipeline for studying how multi-agent systems collaborate under privacy constraints.
The repository is organized as a 3-stage workflow:

1. scenario_generate: Generate benchmark scenarios
2. scenario_simulator: Run experiments with generated scenarios
3. evaluation: Evaluate experiment outputs

## Project Highlights

- End-to-end benchmark pipeline from scenario creation to final scoring
- Privacy-constraint-focused multi-agent collaboration setup
- Reproducible stage-wise execution with modular folders
- Paper-ready scenario set included for simulator experiments

## Repository Layout at a Glance

- scenario_generate: scenario and requirement/constraint generation pipeline
- scenario_simulator: experiment runner, simulator, and agent service
- evaluation: evaluation scripts and result analysis modules

If you prefer isolated environments per stage, you can still install each stage-specific requirements file separately.

## 3-Stage Benchmark Flow

### Environment Setup (Conda)

Create one shared environment at the project root and install unified dependencies.

```bash
conda create -n pac-bench python=3.11 -y
conda activate pac-bench

pip install -r requirements.txt
```

### Stage 1. Build Benchmark Scenarios

Move into scenario_generate and run generation scripts in order.

> [!IMPORTANT]
> Stage 1 is optional if you want to reproduce the paper setting quickly.
> You can directly use the 100 scenarios already included in scenario_simulator/scenarios.
> For broader coverage, refer to the full 1478-scenario dataset on Hugging Face:
> https://huggingface.co/datasets/your-org/pac-bench

Each script writes to a timestamped directory by default and the next script reads that output.
Output pattern:

- result/1_scenario/<timestamp>/<domain>/scenario_<index>.json
- result/2_requirements/<timestamp>/<domain>/requirements_<index>.json
- result/3_memory/<timestamp>/<domain>/memory_<index>.json
- result/4_constraint/<timestamp>/<domain>/constraint_<index>.json

First move to the ```scenario_generate``` directory.
```bash
cd scenario_generate
```

Then generate scenarios by following belows.
```python
# 1) Domain -> Scenario
python 1_run_scenario.py \
	--domain_file domain.json \
	--output result/1_scenario

# 2) Scenario -> Requirements
python 2_run_requirements.py \
	--input result/1_scenario/<output_folder> \
	--output result/2_requirements

# 3) Requirements -> Memory
python 3_run_memory.py \
	--input result/2_requirements/<output_folder> \
	--output result/3_memory

# 4) Memory -> Privacy Constraint
python 4_run_constraint.py \
	--input result/3_memory/<output_folder> \
	--output result/4_constraint
```

This stage produces benchmark scenario artifacts used downstream.

### Stage 2. Simulate Multi-Agent Experiments

Move into scenario_simulator and run the simulator pipeline in this order.

1) Build Docker image
2) Start containers (default: 20 containers)
3) Run parallel experiments
4) Stop containers after experiments

First move to the ```scenario_simulator``` directory.
```bash
cd scenario_simulator
```

Then you can simulate your agents by following belows.
```python
# 1) Build image
bash scripts/build.sh

# 2) Start containers (default: 20)
bash scripts/up.sh

# 3) Run experiments in parallel
bash scripts/run_parallel.sh

# 4) Tear down containers after completion
bash scripts/down.sh
```

> [!CAUTION]
> If you generated new scenarios in Stage 1, copy/use them under scenario_simulator/scenarios and update SCENARIO_ROOT in scripts/run_parallel.sh to the correct path.
>
> If you changed container count or port ranges in scripts/up.sh, you must apply the same changes in scripts/run_parallel.sh (MAX_PARALLEL, SESSION_RANGES, PORT_A_RANGES, PORT_B_RANGES).
>
> If you run experiments with open-source models, you can route models through vLLM or OpenRouter. In that case, you may need to update model configuration files under [scenario_simulator/agent_service/utils](scenario_simulator/agent_service/utils).

For paper experiments, 100 scenarios are provided in the scenarios directory:

- scenario_simulator/scenarios

### Stage 3. Evaluate Experimental Results

Move into evaluation and run the evaluator on simulation outputs.

Before running evaluation, copy the experiment output folder from Stage 2 into evaluation/input.

You can change evaluation options by editing [evaluation/configs/settings.py](evaluation/configs/settings.py).
Key options (examples):

- LLM_PROVIDER, LLM_MODEL: evaluator LLM backend/model selection
- PIPELINE_MAX_WORKERS: parallel evaluation workers
- LOG_LEVEL: logging verbosity

Evaluator toggles in [evaluation/configs/settings.py](evaluation/configs/settings.py):

- EVAL_TASK_ENABLED: enable/disable task evaluation
- EVAL_PRIVACY_ENABLED: enable/disable privacy evaluation
- EVAL_HALLUCINATION_ENABLED: enable/disable hallucination evaluation

First move to the ```evaluation``` directory.
```bash
cd evaluation
```
Then evaluate experiment results.
```python
python main.py --input input/<folder_name>
```

Evaluation outputs are written to the evaluation result directory.

## Scenario Data Access

- Built-in paper experiment scenarios: 100 scenarios in scenario_simulator/scenarios
- Full benchmark scenario collection: 1478 scenarios on Hugging Face dataset

Hugging Face dataset:

- https://huggingface.co/datasets/PAC-Bench/PAC-Bench

## Minimal Run Order

If you want to run the full benchmark once, use this order:

1. Generate scenarios in scenario_generate
2. Run simulations in scenario_simulator
3. Evaluate outputs in evaluation

## Notes

- Use a dedicated Python environment per stage if dependency isolation is needed.
- Check each stage folder for stage-specific configs and output folders.
