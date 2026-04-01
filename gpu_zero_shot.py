import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def build_zero_shot_pipeline(model_id: str = "facebook/bart-large-mnli"):
    device_type = "CPU"
    device = -1
    dml_device = None

    try:
        import torch_directml
    except ImportError:
        torch_directml = None

    if torch.cuda.is_available():
        device = 0
        device_type = "CUDA"
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = 0
        device_type = "Intel XPU"
        print(f"Using Intel GPU (XPU): {torch.xpu.get_device_name(0)}")
    elif torch_directml is not None:
        dml_device = torch_directml.device()
        device_type = "DirectML (GPU)"
        print("Using DirectML GPU acceleration")
    else:
        print("GPU not detected. Using CPU.")
        print("Install IPEX for Intel GPU: conda install -c intel intel-extension-for-pytorch pytorch")
        print("Or install DirectML: pip install torch-directml")

    print(f"Device: {device_type}\n")

    if dml_device is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model = model.to(dml_device)
        zero_shot = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
        )
    else:
        zero_shot = pipeline(
            "zero-shot-classification",
            model=model_id,
            device=device,
        )

    return zero_shot, device_type


if __name__ == "__main__":
    fiction_categories = ["Fiction", "Nonfiction"]
    pipe, device_type = build_zero_shot_pipeline()
    print(f"Pipeline ready on {device_type}")
    print(f"Labels: {fiction_categories}")
