import os
import torch
from torch.utils.data import DataLoader

from dataset_loader import VOCDataset
from faster_rcnn_model import get_model


# ============================================================
# Project Root
# ============================================================

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

results_folder = os.path.join(
    project_root,
    "results"
)

os.makedirs(results_folder, exist_ok=True)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# ============================================================
# Dataset
# ============================================================

print("\nLoading next training dataset...")

dataset = VOCDataset()

print("Dataset loaded successfully.")
print("Training images:", len(dataset))


# ============================================================
# DataLoader
# ============================================================

def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

print("DataLoader created successfully.")


# ============================================================
# Create Model
# ============================================================

print("\nCreating Faster R-CNN model...")

model = get_model()

# ============================================================
# Load Existing Model - Epoch 3
# ============================================================

previous_model = os.path.join(
    results_folder,
    "model_epoch_3.pth"
)

print("\nLoading previous model:")
print(previous_model)

if not os.path.exists(previous_model):
    raise FileNotFoundError(
        f"Model not found: {previous_model}"
    )

checkpoint = torch.load(
    previous_model,
    map_location=device
)

model.load_state_dict(checkpoint)

print("Model Epoch 3 loaded successfully!")

model.to(device)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.002,
    momentum=0.9,
    weight_decay=0.0005
)

print("Optimizer created.")


# ============================================================
# Continue Training
# ============================================================

num_epochs = 2

print("\n==========================================")
print("CONTINUING TRAINING")
print("Starting from: Epoch 3")
print("Next epochs: 4 and 5")
print("==========================================\n")


for epoch in range(num_epochs):

    actual_epoch = epoch + 4

    print(
        f"\n========== Epoch {actual_epoch}/5 =========="
    )

    model.train()

    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(train_loader):

        images = [
            img.to(device)
            for img in images
        ]

        targets = [
            {
                k: v.to(device)
                for k, v in target.items()
            }
            for target in targets
        ]

        # ----------------------------------------
        # Forward pass
        # ----------------------------------------

        loss_dict = model(
            images,
            targets
        )

        losses = sum(
            loss for loss in loss_dict.values()
        )

        # ----------------------------------------
        # Backpropagation
        # ----------------------------------------

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        total_loss += losses.item()

        # ----------------------------------------
        # Display progress
        # ----------------------------------------

        if batch_idx % 100 == 0:

            print(
                f"Batch {batch_idx}/{len(train_loader)} "
                f"| Loss: {losses.item():.4f}"
            )


    # ========================================================
    # Epoch Result
    # ========================================================

    print(
        f"\nEpoch {actual_epoch} completed."
    )

    print(
        f"Total Loss: {total_loss:.4f}"
    )


    # ========================================================
    # Save Model
    # ========================================================

    epoch_model = os.path.join(
        results_folder,
        f"model_epoch_{actual_epoch}.pth"
    )

    torch.save(
        model.state_dict(),
        epoch_model
    )

    print(
        f"Saved: {epoch_model}"
    )


# ============================================================
# Finished
# ============================================================

print("\n==========================================")
print("CONTINUED TRAINING COMPLETED")
print("==========================================")

print("Model 4 and Model 5 saved in results folder.")