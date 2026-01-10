import capybara as cb

cur_folder = cb.get_curdir(__file__)

try:
    import torch

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, 2, 1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 3, 2, 1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
    )
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (dummy_input,),
        cur_folder / "model_shape=224x224.onnx",
        input_names=["input"],
        output_names=["output"],
    )
    torch.onnx.export(
        model,
        (dummy_input,),
        cur_folder / "model_dynamic-axes.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "b", 2: "h", 3: "w"},
            "output": {0: "b", 2: "h", 3: "w"},
        },
    )
except ImportError:
    print("PyTorch is not installed.")
