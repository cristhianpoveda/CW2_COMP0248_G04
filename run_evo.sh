#!/bin/bash

# Define the base results directory
RESULTS_DIR="results"

# Check if the results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Directory '$RESULTS_DIR' does not exist. Run your evaluations first!"
    exit 1
fi

echo "Starting EVO Evaluation Pipeline..."
echo "====================================="

# Loop through all sequence folders inside results/ (e.g., Sequence_A, Sequence_B)
for SEQ_DIR in "$RESULTS_DIR"/Sequence_*; do
    # Skip if it's not a directory
    [ -d "$SEQ_DIR" ] || continue

    SEQ_NAME=$(basename "$SEQ_DIR")
    GT_FILE="$SEQ_DIR/groundtruth.tum"
    TRAJ_DIR="$SEQ_DIR/trajectories"
    EVO_OUT_DIR="$SEQ_DIR/evo_eval"
    
    # Define the single master log file for this sequence
    LOG_FILE="$EVO_OUT_DIR/all_metrics_log.txt"

    echo -e "\nProcessing: $SEQ_NAME"

    # Check if ground truth exists
    if [ ! -f "$GT_FILE" ]; then
        echo "  [WARNING] No groundtruth.tum found in $SEQ_DIR. Skipping..."
        continue
    fi

    # Check if trajectories folder exists
    if [ ! -d "$TRAJ_DIR" ]; then
        echo "  [WARNING] No trajectories folder found in $SEQ_DIR. Skipping..."
        continue
    fi

    # Create output directory for evo results
    mkdir -p "$EVO_OUT_DIR"
    
    # Initialize/Clear the master log file with a header
    echo "==================================================" > "$LOG_FILE"
    echo " EVO Evaluation Results: $SEQ_NAME " >> "$LOG_FILE"
    echo "==================================================" >> "$LOG_FILE"

    # Loop through all .tum files in the trajectories folder
    for TRAJ_FILE in "$TRAJ_DIR"/*.tum; do
        # Handle case where no .tum files exist
        [ -e "$TRAJ_FILE" ] || continue

        TRAJ_NAME=$(basename "$TRAJ_FILE" .tum)
        echo "  -> Evaluating: $TRAJ_NAME"
        
        # Add a sub-header for this specific trajectory in the log file
        echo -e "\n\n--------------------------------------------------" >> "$LOG_FILE"
        echo " Trajectory: $TRAJ_NAME " >> "$LOG_FILE"
        echo "--------------------------------------------------" >> "$LOG_FILE"

        # 1. Run Absolute Pose Error (APE)
        echo -e "\n>>> Absolute Pose Error (APE)" >> "$LOG_FILE"
        evo_ape tum "$GT_FILE" "$TRAJ_FILE" -a \
            --save_plot "$EVO_OUT_DIR/${TRAJ_NAME}_ape_plot.png" \
            >> "$LOG_FILE" 2>&1

        # 2. Run Relative Pose Error (RPE)
        echo -e "\n>>> Relative Pose Error (RPE)" >> "$LOG_FILE"
        evo_rpe tum "$GT_FILE" "$TRAJ_FILE" -a \
            --save_plot "$EVO_OUT_DIR/${TRAJ_NAME}_rpe_plot.png" \
            >> "$LOG_FILE" 2>&1

    done
    
    echo "  [SUCCESS] Evaluations and plots saved to $EVO_OUT_DIR"
    echo "  [INFO] Master metrics log saved to $LOG_FILE"
done

echo -e "\n====================================="
echo "All sequences evaluated successfully!"