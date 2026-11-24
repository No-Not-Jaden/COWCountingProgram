from ultralytics import YOLO
import torch

# https://github.com/ultralytics/ultralytics
if __name__ == "__main__":

    assert torch.cuda.is_available(), "CUDA is unavailable!"

    print("Training on:", torch.cuda.get_device_name(0))

    model = YOLO("yolo26n.pt")

    results = model.train(
        data="dataset/cow_dataset/cows.yaml",

        epochs=100,
        imgsz=640,
        batch=8,

        device=0,
        workers=4,

        amp=True,
        patience=15,

        project="cow_training",
        name="ground_level_baseline",
        exist_ok=True
    )

    # Evaluate on the test dataset
    model = YOLO(
        "runs/detect/cow_training/ground_level_baseline/weights/best.pt"
    )

    metrics = model.val(
        data="dataset/cow_dataset/cows.yaml",
        split="test",
        device=0
    )


    # Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model
    print("Exported Model Path: ", path)