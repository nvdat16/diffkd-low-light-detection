# DiffKD-Yolo

DiffKD-Yolo is a **knowledge distillation (KD)** project for **object detection** built on the **Ultralytics YOLO** ecosystem. It uses a **teacher model** and a **student model** to improve the student through **feature distillation** with the **DiffKD** module.

## Overview

Main training pipeline:

1. Load the **YOLO student** from a checkpoint or default configuration.
2. Load the **teacher model** from a checkpoint.
3. Hook feature layers between the teacher and the student.
4. Compute **feature loss** with `DiffKD` to bring the student closer to the teacher.
5. Train the detector using the custom trainer in `engine/`.

## Features

- Detection training with **knowledge distillation**.
- Feature hooking between teacher and student.
- Integrated **DiffKD** module in `models/diffkd.py`.
- Support for different teacher architectures with hook adjustments.
- Compatible with the **Ultralytics** training workflow.

## Project Structure

```text
DiffKD-Yolo/
├── train.py                # Training entry point
├── config.yaml             # Config file (currently empty)
├── engine/
│   ├── base.py             # Base trainer + distillation logic
│   └── trainer.py          # Custom DetectionTrainer
├── models/
│   ├── diffkd.py           # DiffKD module
│   ├── irformer.py         # IRFormer teacher model
│   └── mbllen.py           # MBLLEN teacher model
├── dataset/                # Data
└── experiments/
    ├── checkpoints/        # Saved checkpoints
    └── logs/               # Training logs
```

## Outputs

During training, outputs are usually saved in:

- `experiments/checkpoints/`
- `experiments/logs/`


## Requirements

- Python 3.10+ recommended
- PyTorch
- Ultralytics
- NumPy

If you are using CUDA, install the PyTorch build that matches your GPU and driver setup.

## Installation

```bash
pip install ultralytics torch torchvision numpy einops
```

If you already have a virtual environment, activate it before installing dependencies.

## Dataset Setup

The project uses a dataset config file in the Ultralytics format, for example `data/coco.yaml`.

You can replace it with your own YAML file:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 1
names: ["class_name"]
```

## Training

Basic training command:

```bash
python train.py --teacher /path/to/teacher.pt --model yolov10s.pt --data data/coco.yaml --epochs 50 --batch 16
```

### Command-line arguments

- `--teacher`: path to the teacher checkpoint, required.
- `--model`: student checkpoint, default `yolov10s.pt`.
- `--data`: dataset config file, default `data/coco.yaml`.
- `--epochs`: number of training epochs, default `50`.
- `--batch`: batch size, default `16`.

## Important Notes

- `engine/base.py` contains the main distillation logic.
- `engine/trainer.py` hooks features by default layer index or by module name, depending on the configuration.
- `models/diffkd.py` defines the diffusion pipeline used for feature distillation.
- `models/mbllen.py` is the default teacher in `build_teacher_model()`.
- If you want to use another teacher such as `IRFormer`, update `build_teacher_model()` and adjust the hook layers if needed.

## Teacher/Student Configuration

In `engine/base.py`, the `DistillationTrainer` class supports:

- custom layer-name hooking
- Ultralytics default-structure hooking
- feature transfer between teacher and student for distillation loss computation


## Author

- Full name: Nguyen Van Dat
- Email: nvdat1601@gmai.com
