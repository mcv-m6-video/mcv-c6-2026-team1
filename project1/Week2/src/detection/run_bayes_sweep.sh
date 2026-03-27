#!/bin/bash

# 1. Capture the target YAML file and number of runs from the console
CONFIG_FILE=$1
NUM_RUNS=$2

# Validate that an argument is provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Missing configuration file."
    echo "Usage: src/detection/run_bayes_sweep.sh <path_to_yaml>"
    echo "Example: src/detection/run_bayes_sweep.sh src/detection/sweep_bayes_yolo.yaml"
    exit 1
fi

# Create a master logs directory
BASE_LOG_DIR="src/detection/logs"
mkdir -p "$BASE_LOG_DIR"

# Extract the pure filename without path or extension (e.g., "sweep_yolo")
BASENAME=$(basename "$CONFIG_FILE" .yaml)

# Create a specific sub-folder for this model
MODEL_LOG_DIR="$BASE_LOG_DIR/$BASENAME"
mkdir -p "$MODEL_LOG_DIR"

# Create a timestamped log file for this specific execution
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$MODEL_LOG_DIR/execution_${TIMESTAMP}.log"

echo "================================================="
echo "Initializing Sweep from: $CONFIG_FILE"
echo "Detailed training logs route to: $LOG_FILE"
echo "================================================="

# 2. Create the sweep and capture the terminal output
SWEEP_OUTPUT=$(wandb sweep "$CONFIG_FILE" 2>&1)

# 3. Parse the output to find the exact 'wandb agent' command
AGENT_CMD=$(echo "$SWEEP_OUTPUT" | grep -oE "wandb agent [^[:space:]]+")

if [ -z "$AGENT_CMD" ]; then
    echo "Error: Could not extract the wandb agent command from the output."
    echo "Raw output:"
    echo "$SWEEP_OUTPUT"
    exit 1
fi

# Append the --count flag to the agent command
if [ -n "$NUM_RUNS" ]; then
    echo "Applying limit: Agent will stop after $NUM_RUNS runs."
    AGENT_CMD="$AGENT_CMD --count $NUM_RUNS"
fi

echo "Sweep successfully created! Launching agent..."
echo "Executing: $AGENT_CMD"

# 4. Run the agent and redirect ALL its heavy output directly into the log file
eval "$AGENT_CMD" > "$LOG_FILE" 2>&1

echo "Sweep $CONFIG_FILE is complete."
echo "Review the execution logs at: $LOG_FILE"
echo ""