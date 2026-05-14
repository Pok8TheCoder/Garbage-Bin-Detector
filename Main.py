"""
YOLOv9 Garbage Bin Detection & Segmentation Training Script
Optimized for RTX 3060 12GB GPU
"""

import os
import torch
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────
# Task Selection: 'detect' or 'segment'
TASK = 'detect' 

# YOLOv9 Model Selection:
# Detection: yolov9t.pt, yolov9s.pt, yolov9m.pt, yolov9c.pt, yolov9e.pt
# Segmentation: yolov9c-seg.pt, yolov9e-seg.pt
BASE_MODEL = "yolov9s.pt" if TASK == 'detect' else "yolov9c-seg.pt"

# Dataset Path
DATASET_YAML = "FinalDataset/data.yaml" if TASK == 'detect' else "MaskedDataset/data.yaml"

# Hardware Settings (Optimized for RTX 3060 12GB)
IMG_SIZE = 640
BATCH_SIZE = 48 if TASK == 'detect' else 16  # Segmentation requires more VRAM
EPOCHS = 200
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ─── Environment Setup ────────────────────────────────────────────────────────
def setup_env():
    print(f"🚀 Initializing YOLOv9 {TASK.upper()} Pipeline...")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Ensure runs directory exists
    os.makedirs("runs/v9", exist_ok=True)

def train():
    # Load YOLOv9 model
    print(f"\n🆕 Loading base model: {BASE_MODEL}...")
    model = YOLO(BASE_MODEL)

    # Start Training
    print(f"\n🔥 Starting YOLOv9 {TASK} training on {DATASET_YAML}...")
    results = model.train(
        # --- [ CORE SETTINGS ] ---
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=4,
        name=f"garbage_bin_{TASK}_v9",
        project=os.path.join(os.getcwd(), f"runs/v9/{TASK}"),
        exist_ok=True,
        optimizer="auto",
        verbose=True,
        
        # --- [ TRAINING MODES ] ---
        resume=False,                  # Start fresh for v9
        save=True,
        save_period=5,
        patience=50,
        amp=True,                      # Use Automatic Mixed Precision
        compile=True,                  # Optimize for speed (PyTorch 2.0+)

        # --- [ HYPERPARAMETERS (OPTIMIZED FOR v9) ] ---
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        
        # --- [ LOSS GAINS ] ---
        box=7.5,                       # Box loss gain
        cls=0.5,                       # Class loss gain
        dfl=1.5,                       # DFL loss gain
        
        # --- [ AUGMENTATION ] ---
        augment=True,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        flipud=0.0,
        fliplr=0.5,
        degrees=10.0,
    )

    print("\n✅ Training Complete!")
    best_path = os.path.join(results.save_dir, "weights/best.pt")
    print(f"Best weights saved at: {best_path}")
    
    return model

def validate(model):
    print("\n📊 Running Validation...")
    metrics = model.val()
    print(f"Validation Results: {metrics}")

def test_inference(model):
    # Determine test path based on dataset
    test_img_dir = "FinalDataset/test/images" if TASK == 'detect' else "MaskedDataset/images"
    
    if os.path.exists(test_img_dir):
        print(f"\n🧪 Running Inference on {test_img_dir}...")
        model.predict(
            source=test_img_dir,
            save=True,
            project=f"runs/v9/{TASK}",
            name="test_results",
            conf=0.25,
        )
    else:
        print(f"\n⚠ Skip inference: {test_img_dir} not found.")

if __name__ == "__main__":
    setup_env()
    trained_model = train()
    validate(trained_model)
    test_inference(trained_model)
    print("\n🎉 YOLOv9 Pipeline Complete!")

