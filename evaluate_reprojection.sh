#!/bin/bash

SEQUENCES=("Sequence_A" "Sequence_B")

echo "Image Reprojection Evaluation..."
echo "================================================="

# Loop through each sequence
for SEQ in "${SEQUENCES[@]}"; do
    SEQ_PATH="data/$SEQ/camera_color_image_raw"
    
    echo -e "\n>>> Processing: $SEQ"
    echo "-------------------------------------------------"
    
    # Forward Warping with Splatting (Defaults)
    echo "  [1/3] Running: Forward Warping + Splatting"
    python -m image_reprojection.reproject_images --sequence_path "$SEQ_PATH"
    
    # Forward Warping without Splatting
    echo "  [2/3] Running: Forward Warping + No Splatting"
    python -m image_reprojection.reproject_images --sequence_path "$SEQ_PATH" --no_splatting
    
    # Inverse Warping
    echo "  [3/3] Running: Inverse Warping"
    python -m image_reprojection.reproject_images --sequence_path "$SEQ_PATH" --warping inverse
    
done

echo -e "\n================================================="
echo "All tests completed successfully!"