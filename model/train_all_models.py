
import os
import sys
import csv
import torch
from torch.utils.data import DataLoader

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RESULTS_FOLDER = os.path.join(
    PROJECT_ROOT, "results", "retrained"
)

os.makedirs(RESULTS_FOLDER, exist_ok=True)

LOSS_HISTORY_FILE = os.path.join(
    RESULTS_FOLDER,
    "training_history.csv"
)

# Make project model modules importable
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import VOCDataset
from faster_rcnn_model import get_model


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 2
NUM_CLASSES = 10

# One training pass per dataset group
NUM_EPOCHS_PER_MODEL = 1

DATASET_RANGES = {
    1: (0, 2000),
    2: (2000, 4000),
    3: (4000, 6000),
    4: (6000, 8000),
    5: (8000, 10000),
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("FIVE-MODEL SEQUENTIAL RETRAINING")
print("=" * 60)
print("Project:", PROJECT_ROOT)
print("Device:", device)
print("Output:", RESULTS_FOLDER)
print("Loss history:", LOSS_HISTORY_FILE)


# --------------------------------------------------
# COLLATE FUNCTION
# --------------------------------------------------

def collate_fn(batch):
    return tuple(zip(*batch))


# --------------------------------------------------
# LOSS HISTORY INITIALIZATION
# --------------------------------------------------

def initialize_loss_history():
    """Create a fresh CSV for this training run."""

    with open(
        LOSS_HISTORY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "model",
            "epoch",
            "training_loss"
        ])


def save_training_loss(model_number, epoch, average_loss):
    """Save the measured average training loss."""

    with open(
        LOSS_HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            model_number,
            epoch,
            average_loss
        ])


# --------------------------------------------------
# TRAINING FUNCTION
# --------------------------------------------------

def train_one_model(model, train_loader, model_number):

    model.train()

    optimizer = torch.optim.SGD(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    total_batches = len(train_loader)

    for epoch in range(NUM_EPOCHS_PER_MODEL):

        total_loss = 0.0

        for batch_index, (images, targets) in enumerate(
            train_loader, start=1
        ):

            images = [
                image.to(device)
                for image in images
            ]

            targets = [
                {
                    key: value.to(device)
                    for key, value in target.items()
                }
                for target in targets
            ]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss in Model {model_number}, "
                    f"batch {batch_index}: {loss.item()}"
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_index % 50 == 0 or batch_index == total_batches:
                print(
                    f"Model {model_number} | "
                    f"Epoch {epoch + 1}/{NUM_EPOCHS_PER_MODEL} | "
                    f"Batch {batch_index}/{total_batches} | "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = total_loss / max(total_batches, 1)

        # Save the actual measured loss
        save_training_loss(
            model_number,
            epoch + 1,
            average_loss
        )

        print(
            f"Model {model_number} | "
            f"Epoch {epoch + 1} completed | "
            f"Average loss: {average_loss:.4f}"
        )


# --------------------------------------------------
# MAIN TRAINING PIPELINE
# --------------------------------------------------

def main():

    # Protect existing checkpoints
    expected_files = [
        os.path.join(
            RESULTS_FOLDER,
            f"model_epoch_{number}.pth"
        )
        for number in range(1, 6)
    ]

    existing_files = [
        path for path in expected_files
        if os.path.exists(path)
    ]

    if existing_files:
        raise FileExistsError(
            "Existing retrained checkpoints were found. "
            "Training was stopped to protect them.\n"
            + "\n".join(existing_files)
        )

    # Verify dataset availability
    image_folder = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "train",
        "enhanced",
    )

    if not os.path.isdir(image_folder):
        raise FileNotFoundError(
            f"Training image folder not found: {image_folder}"
        )

    all_images = sorted([
        name for name in os.listdir(image_folder)
        if name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    if len(all_images) < 10000:
        raise ValueError(
            f"Expected at least 10,000 images, "
            f"but found {len(all_images)}."
        )

    print("\nTotal available images:", len(all_images))

    # Start a fresh history for this new run
    initialize_loss_history()

    previous_checkpoint = None

    for model_number, (start_index, end_index) in (
        DATASET_RANGES.items()
    ):

        print("\n" + "=" * 60)
        print(f"STARTING MODEL {model_number}")
        print("=" * 60)

        dataset = VOCDataset(
            start_index=start_index,
            end_index=end_index,
        )

        if len(dataset) != 2000:
            raise ValueError(
                f"Model {model_number} requires 2,000 images; "
                f"found {len(dataset)}."
            )

        train_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

        model = get_model()

        if previous_checkpoint is not None:

            print(
                "Loading previous newly trained checkpoint:",
                previous_checkpoint,
            )

            checkpoint = torch.load(
                previous_checkpoint,
                map_location=device,
                weights_only=True,
            )

            model.load_state_dict(checkpoint)

        else:
            print(
                "Model 1 starts from get_model() initialization."
            )
            print(
                "Verify that get_model() uses the intended "
                "pretrained Faster R-CNN weights."
            )

        model.to(device)

        train_one_model(
            model,
            train_loader,
            model_number,
        )

        output_path = os.path.join(
            RESULTS_FOLDER,
            f"model_epoch_{model_number}.pth",
        )

        torch.save(
            model.state_dict(),
            output_path,
        )

        print("Saved:", output_path)

        previous_checkpoint = output_path

        del model
        del train_loader
        del dataset

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("ALL FIVE MODELS HAVE FINISHED TRAINING")
    print("=" * 60)

    print("Training history saved to:", LOSS_HISTORY_FILE)

    for model_number in range(1, 6):

        checkpoint_path = os.path.join(
            RESULTS_FOLDER,
            f"model_epoch_{model_number}.pth",
        )

        print(
            f"Model {model_number}: {checkpoint_path}"
        )


if __name__ == "__main__":
    main()