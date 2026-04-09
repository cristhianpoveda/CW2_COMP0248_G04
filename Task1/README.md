# CW2 Task 1: Hand Gesture Recognition from Point Clouds

10-class gesture classification using depth-only data converted to 3D point clouds.
Uses a Random Forest pixel classifier (trained on CW1 annotation masks) to detect
hands in depth images. At inference the pipeline is depth-only.

## Setup

```
pip install -r requirements.txt
```

## Step 0: Train hand detector (one time)
```
python -m src.hand_detector --data_root ../rgb_depth --output hand_detector.pkl
```

## Step 1: Preprocess depth -> point clouds
```
python -m src.preprocess --data_root ../rgb_depth --test_root ../test_data --detector hand_detector.pkl
```

## Step 2: Train models
```
python -m src.train --model pointnet
python -m src.train --model dgcnn
```

## Step 3: Evaluate
Validation:
```
python -m src.evaluate --checkpoint weights/pointnet.pth --model pointnet
python -m src.evaluate --checkpoint weights/dgcnn.pth --model dgcnn
```
Test:
```
python -m src.evaluate --data_root data/test_pc --checkpoint weights/pointnet.pth --model pointnet --test
python -m src.evaluate --data_root data/test_pc --checkpoint weights/dgcnn.pth --model dgcnn --test
```

## Step 4: Figures
```
python -m src.visualise
python -m src.visualise --test
```

## Models

| Model | Type | Description |
|-------|------|-------------|
| PointNet | Baseline | Global feature extraction with spatial transformers |
| DGCNN | Advanced | Dynamic graph convolution with spatial transformer |
