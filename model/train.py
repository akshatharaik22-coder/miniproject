import torch
from torch.utils.data import DataLoader

from dataset_loader import VOCDataset
from faster_rcnn_model import get_model

# ===========================
# Device
# ===========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ===========================
# Create Dataset
# ===========================

print("\nStep 1: Loading Dataset...")

dataset = VOCDataset()

print("Dataset Loaded Successfully")
print("Total Images:", len(dataset))

# ===========================
# DataLoader
# ===========================

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

print("DataLoader Created Successfully")

# ===========================
# Load Model
# ===========================

print("\nStep 3: Loading Faster R-CNN Model...")

model = get_model()

print("Model Loaded Successfully")

model.to(device)

# ===========================
# Optimizer
# ===========================

print("\nStep 4: Creating Optimizer...")

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

print("Optimizer Created Successfully")

# ===========================
# Training
# ===========================

num_epochs = 2

print("\nStep 5: Starting Training...\n")

for epoch in range(num_epochs):

    print(f"\nEpoch {epoch+1}/{num_epochs}")

    model.train()

    total_loss = 0

    for batch_idx, (images, targets) in enumerate(train_loader):

        images = [img.to(device) for img in images]

        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        total_loss += losses.item()

        if batch_idx % 100 == 0:
            print(
                f"Batch {batch_idx}/{len(train_loader)}  Loss: {losses.item():.4f}"
            )

    print(f"Epoch {epoch+1} Completed")
    print(f"Total Loss: {total_loss:.4f}")

# ===========================
# Save Model
# ===========================

torch.save(model.state_dict(), "saved_model.pth")

print("\nTraining Completed Successfully!")
print("Model Saved as saved_model.pth")