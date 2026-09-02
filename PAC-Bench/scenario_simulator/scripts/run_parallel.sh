#!/usr/bin/env bash
set -eo pipefail

# ============================================================================
# Parallel Run Script - Execute multiple run.sh instances in parallel
# ============================================================================

# Run from repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- Configuration ----
SCENARIO_ROOT="scenarios/change"
RESULT_ROOT="results"

# Define domains to test
DOMAINS=(
    "Energy_Equipment_and_Services"
	"Oil_Gas_and_Consumable_Fuels"
	"Chemicals"
	"Construction_Materials"
	"Containers_and_Packaging"
	"Metals_and_Mining"
	"Paper_and_Forest_Products"
	"Aerospace_and_Defense"
	"Building_Products"
	"Construction_and_Engineering"
	"Electrical_Equipment"
	"Machinery"
	"Trading_Companies_and_Distributors"
	"Commercial_Services_and_Supplies"
	"Air_Freight_and_Logistics"
	"Passenger_Airlines"
	"Marine_Transportation"
	"Ground_Transportation"
	"Transportation_Infrastructure"
	"Automobiles"
	"Leisure_Products"
	"Textiles_Apparel_and_Luxury_Goods"
	"Hotels_Restaurants_and_Leisure"
	"Diversified_Consumer_Services"
	"Broadline_Retail"
	"Consumer_Staples_Distribution_and_Retail"
	"Food_Products"
	"Household_Products"
	"Health_Care_Providers_and_Services"
	"Biotechnology"
	"Pharmaceuticals"
	"Life_Sciences_Tools_and_Services"
	"Banks"
	"Financial_Services"
	"Capital_Markets"
	"Mortgage_Real_Estate_Investment_Trusts_(REITs)"
	"Insurance"
	"IT_Services"
	"Software"
	"Communications_Equipment"
	"Technology_Hardware_Storage_and_Peripherals"
	"Semiconductors_and_Semiconductor_Equipment"
	"Wireless_Telecommunication_Services"
	"Media"
	"Entertainment"
	"Electric_Utilities"
	"Gas_Utilities"
	"Water_Utilities"
	"Independent_Power_and_Renewable_Electricity_Producers"
	"Real_Estate_Management_and_Development"
)

# Define models to test
MODELS=(
    "gpt-5.1"
    "claude-sonnet-4-5"
	"llama3-70b"
	"qwen3-32b"
	# "qwen3-8b"
)

# Maximum parallel jobs
MAX_PARALLEL=10

# Session and port configuration (10 slots for parallel execution)
# Each slot gets 2 sessions and 2 ports
SESSION_RANGES=(
    "[0,2)"   "[2,4)"   "[4,6)"   "[6,8)"   "[8,10)"
    "[10,12)" "[12,14)" "[14,16)" "[16,18)" "[18,20)"
)
PORT_A_RANGES=(
    "[8100,8102)" "[8102,8104)" "[8104,8106)" "[8106,8108)" "[8108,8110)"
    "[8110,8112)" "[8112,8114)" "[8114,8116)" "[8116,8118)" "[8118,8120)"
)
PORT_B_RANGES=(
    "[8200,8202)" "[8202,8204)" "[8204,8206)" "[8206,8208)" "[8208,8210)"
    "[8210,8212)" "[8212,8214)" "[8214,8216)" "[8216,8218)" "[8218,8220)"
)

# Additional run.sh parameters
MAX_TURNS=20
MAX_STEPS=10
MAX_TOOL_CALLS=5

# ============================================================================
# Setup
# ============================================================================

# Create timestamp for this batch run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_RESULT_DIR="${RESULT_ROOT}/${TIMESTAMP}"
mkdir -p "${BATCH_RESULT_DIR}"

echo "============================================================================"
echo "Parallel Execution Started: ${TIMESTAMP}"
echo "============================================================================"
echo "Scenario Root: ${SCENARIO_ROOT}"
echo "Result Root:   ${BATCH_RESULT_DIR}"
echo "Max Parallel:  ${MAX_PARALLEL}"
echo "Domains:       ${DOMAINS[*]}"
echo "Models:        ${MODELS[*]}"
echo "============================================================================"

# ============================================================================
# Process Management
# ============================================================================

# Track background jobs
declare -a PIDS=()
declare -a SLOT_AVAILABLE=()

# Initialize slot availability (1 = available, 0 = in use) and PIDS
for i in $(seq 0 $((MAX_PARALLEL - 1))); do
    SLOT_AVAILABLE[$i]=1
    PIDS[$i]=""
done

# Cleanup function - kills all child processes
cleanup() {
    echo ""
    echo "============================================================================"
    echo "Caught interrupt signal - Cleaning up..."
    echo "============================================================================"
    
    # Kill all background jobs
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "Terminating ${#PIDS[@]} running processes..."
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Killing process $pid and its children..."
                # Kill the process group to ensure all children are terminated
                pkill -TERM -P "$pid" 2>/dev/null || true
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
        
        # Wait a bit and force kill if needed
        sleep 2
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force killing process $pid..."
                pkill -KILL -P "$pid" 2>/dev/null || true
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
    fi
    
    echo "Cleanup complete."
    exit 1
}

# Set trap for Ctrl+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

# ============================================================================
# Job Management Functions
# ============================================================================

# Find an available slot
find_available_slot() {
    for i in $(seq 0 $((MAX_PARALLEL - 1))); do
        if [ "${SLOT_AVAILABLE[$i]}" -eq 1 ]; then
            echo "$i"
            return 0
        fi
    done
    echo "-1"
}

# Wait for any job to complete and return the freed slot
wait_for_slot() {
    while true; do
        local found_slot=0
        for i in $(seq 0 $((MAX_PARALLEL - 1))); do
            # Check if this slot has a PID
            if [ -n "${PIDS[$i]:-}" ]; then
                pid="${PIDS[$i]}"
                if ! kill -0 "$pid" 2>/dev/null; then
                    # Process completed
                    wait "$pid" 2>/dev/null || true
                    SLOT_AVAILABLE[$i]=1
                    PIDS[$i]=""
                    echo "$i"
                    return 0
                fi
            fi
        done
        sleep 1
    done
}

# ============================================================================
# Build Job Queue
# ============================================================================

declare -a JOB_QUEUE=()

for domain in "${DOMAINS[@]}"; do
    scenario_dir="${SCENARIO_ROOT}/${domain}"
    
    # Check if scenario directory exists
    if [ ! -d "$scenario_dir" ]; then
        echo "Warning: Scenario directory not found: $scenario_dir"
        continue
    fi
    
    # Test all model pairs
    for model_a in "${MODELS[@]}"; do
        for model_b in "${MODELS[@]}"; do
            # Create result directory for this combination
            result_subdir="${BATCH_RESULT_DIR}/${domain}/${model_a}_${model_b}"
            mkdir -p "$result_subdir" || {
                echo "Error: Failed to create directory: $result_subdir"
                continue
            }
            
            # Add job to queue: "domain|model_a|model_b|scenario_dir|result_dir"
            JOB_QUEUE+=("${domain}|${model_a}|${model_b}|${scenario_dir}|${result_subdir}")
        done
    done
done

# Check if we have any jobs to execute
if [ ${#JOB_QUEUE[@]} -eq 0 ]; then
    echo "Error: No valid jobs to execute. Exiting."
    exit 1
fi

echo "Total jobs to execute: ${#JOB_QUEUE[@]}"
echo "============================================================================"
echo ""

# ============================================================================
# Execute Jobs in Parallel
# ============================================================================

job_index=0
total_jobs=${#JOB_QUEUE[@]}

for job in "${JOB_QUEUE[@]}"; do
    # Parse job
    IFS='|' read -r domain model_a model_b scenario_dir result_dir <<< "$job"
    
    # Find available slot
    slot=$(find_available_slot)
    if [ "$slot" -eq -1 ]; then
        # Wait for a slot to become available
        slot=$(wait_for_slot)
    fi
    
    # Mark slot as in use
    SLOT_AVAILABLE[$slot]=0
    
    # Get session and port ranges for this slot
    session_range="${SESSION_RANGES[$slot]}"
    port_a_range="${PORT_A_RANGES[$slot]}"
    port_b_range="${PORT_B_RANGES[$slot]}"
    
    # Increment job counter
    job_index=$((job_index + 1))
    
    echo "[Job $job_index/$total_jobs] Starting: $domain | $model_a vs $model_b (Slot $slot)"
    echo "  Session: $session_range | Port-A: $port_a_range | Port-B: $port_b_range"
    echo "  Result: $result_dir"
    
    # Execute run.sh in background
    (
        export SCENARIO_DIR="$scenario_dir"
        export RESULT_DIR="$result_dir"
        export SESSION_RANGE="$session_range"
        export PORT_A_RANGE="$port_a_range"
        export PORT_B_RANGE="$port_b_range"
        export MODEL_AGENT_A="$model_a"
        export MODEL_AGENT_B="$model_b"
        export MAX_TURNS="$MAX_TURNS"
        export MAX_STEPS="$MAX_STEPS"
        export MAX_TOOL_CALLS="$MAX_TOOL_CALLS"
        
        echo "$(date): Job started" > "${result_dir}/job_status.log"
        echo "SCENARIO_DIR=$SCENARIO_DIR" >> "${result_dir}/job_status.log"
        
        bash "${SCRIPT_DIR}/run.sh" >> "${result_dir}/run.log" 2>&1
        
        exit_code=$?
        echo "$(date): Job finished with exit code $exit_code" >> "${result_dir}/job_status.log"
        
        if [ $exit_code -eq 0 ]; then
            echo "[Job $job_index/$total_jobs] ✓ Completed: $domain | $model_a vs $model_b"
        else
            echo "[Job $job_index/$total_jobs] ✗ Failed: $domain | $model_a vs $model_b (exit code: $exit_code)"
        fi
    ) &
    
    # Store PID
    PIDS[$slot]=$!
    echo "  PID: ${PIDS[$slot]}"
    
    # Small delay to avoid overwhelming the system
    sleep 0.5
done

# ============================================================================
# Wait for All Jobs to Complete
# ============================================================================

echo ""
echo "============================================================================"
echo "All jobs dispatched. Waiting for completion..."
echo "============================================================================"

# Wait for all remaining jobs
for i in $(seq 0 $((MAX_PARALLEL - 1))); do
    if [ -n "${PIDS[$i]:-}" ]; then
        wait "${PIDS[$i]}" 2>/dev/null || true
    fi
done

echo ""
echo "============================================================================"
echo "All jobs completed!"
echo "Results saved to: ${BATCH_RESULT_DIR}"
echo "============================================================================"
