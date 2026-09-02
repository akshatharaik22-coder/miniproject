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


os.makedirs(
    results_folder,
    exist_ok=True
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("Using Device:", device)


# ============================================================
# Create Dataset
# ============================================================

print("\nStep 1: Loading Next 2000 Images...")

dataset = VOCDataset()


print(
    "Dataset Loaded Successfully"
)

print(
    "Training Images:",
    len(dataset)
)


# ============================================================
# DataLoader
# ============================================================

print("\nStep 2: Creating DataLoader...")


def collate_fn(batch):

    return tuple(zip(*batch))


train_loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=True,

    collate_fn=collate_fn,

    num_workers=0
)


print(
    "DataLoader Created Successfully"
)


# ============================================================
# Create Model Architecture
# ============================================================

print("\nStep 3: Creating Faster R-CNN Model...")

model = get_model()


# ============================================================
# Load Previous Model
# ============================================================

previous_model_path = os.path.join(

    results_folder,

    "model_epoch_2.pth"
)


print(
    "\nLoading previous model:"
)

print(
    previous_model_path
)


checkpoint = torch.load(

    previous_model_path,

    map_location=device,

    weights_only=False
)


model.load_state_dict(
    checkpoint
)


print(
    "Previous model loaded successfully!"
)


# ============================================================
# Move Model to Device
# ============================================================

model.to(device)


# ============================================================
# Optimizer
# ============================================================

print("\nStep 4: Creating Optimizer...")


optimizer = torch.optim.SGD(

    model.parameters(),

    lr=0.001,

    momentum=0.9,

    weight_decay=0.0005
)


print(
    "Optimizer Created Successfully"
)


# ============================================================
# Training Settings
# ============================================================

# One new epoch = next 2000 images

num_epochs = 1


print(
    "\nStep 5: Starting Continued Training..."
)

print(
    "Previous Model: model_epoch_2.pth"
)

print(
    "New Dataset: Next 2000 Images"
)


# ============================================================
# Training Loop
# ============================================================

for epoch in range(num_epochs):

    print(
        f"\n========== Continued Epoch 3 =========="
    )


    model.train()


    total_loss = 0


    for batch_idx, (images, targets) in enumerate(
        train_loader
    ):


        # ==============================================
        # Move Images to Device
        # ==============================================

        images = [

            img.to(device)

            for img in images

        ]


        # ==============================================
        # Move Targets to Device
        # ==============================================

        targets = [

            {
                k: v.to(device)

                for k, v in t.items()
            }

            for t in targets

        ]


        # ==============================================
        # Forward Pass
        # ==============================================

        loss_dict = model(

            images,

            targets

        )


        # ==============================================
        # Total Loss
        # ==============================================

        losses = sum(

            loss

            for loss in loss_dict.values()

        )


        # ==============================================
        # Clear Gradients
        # ==============================================

        optimizer.zero_grad()


        # ==============================================
        # Backpropagation
        # ==============================================

        losses.backward()


        # ==============================================
        # Update Model
        # ==============================================

        optimizer.step()


        # ==============================================
        # Track Loss
        # ==============================================

        total_loss += losses.item()


        # ==============================================
        # Print Progress
        # ==============================================

        if batch_idx % 100 == 0:

            print(

                f"Batch "
                f"{batch_idx}/{len(train_loader)} "
                f"| Loss: "
                f"{losses.item():.4f}"

            )


    # ==================================================
    # Epoch Completed
    # ==================================================

    print(
        "\nContinued Epoch 3 Completed"
    )


    print(
        f"Total Loss: {total_loss:.4f}"
    )


    # ==================================================
    # Save Epoch 3
    # ==================================================

    epoch_model = os.path.join(

        results_folder,

        "model_epoch_3.pth"

    )


    torch.save(

        model.state_dict(),

        epoch_model

    )


    print(
        "\n===================================="
    )

    print(
        "MODEL EPOCH 3 SAVED"
    )

    print(
        "===================================="
    )

    print(
        epoch_model
    )


# ============================================================
# Training Completed
# ============================================================

print(
    "\n===================================="
)

print(
    "Continued Training Completed!"
)

print(
    "model_epoch_3.pth created successfully!"
)

print(
    "===================================="
)