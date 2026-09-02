import os
import torch

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

model_path = os.path.join(
    project_root,
    "results",
    "model_epoch_2.pth"
)

print("Checking:", model_path)

try:
    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    print("SUCCESS!")
    print("model_epoch_2.pth is valid.")
    print("Number of parameters:", len(checkpoint))

except Exception as e:
    print("ERROR!")
    print(e)