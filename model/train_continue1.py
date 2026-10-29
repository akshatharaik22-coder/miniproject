import os
import sys
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

if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import VOCDataset
from faster_rcnn_model import get_model


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 2
NUM_EPOCHS_PER_MODEL = 1

# Model 4: images 6001–8000
# Model 5: images 8001–10000
DATASET_RANGES = {
    4: (6000, 8000),
    5: (8000, 10000),
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# COLLATE FUNCTION
# --------------------------------------------------

def collate_fn(batch):
    return tuple(zip(*batch))


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

        print(
            f"Model {model_number} completed | "
            f"Average loss: {average_loss:.4f}"
        )


# --------------------------------------------------
# RESUME TRAINING FROM MODEL 3
# --------------------------------------------------

def main():

    print("=" * 60)
    print("RESUME TRAINING: MODEL 4 AND MODEL 5")
    print("=" * 60)

    print("Project:", PROJECT_ROOT)
    print("Device:", device)
    print("Checkpoint folder:", RESULTS_FOLDER)

    # Start from the completed Model 3 checkpoint.
    previous_checkpoint = os.path.join(
        RESULTS_FOLDER,
        "model_epoch_3.pth"
    )

    if not os.path.isfile(previous_checkpoint):
        raise FileNotFoundError(
            "Model 3 checkpoint was not found:\n"
            f"{previous_checkpoint}\n"
            "Correct the checkpoint path before running."
        )

    # Confirm the image folder exists.
    image_folder = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "train",
        "enhanced"
    )

    if not os.path.isdir(image_folder):
        raise FileNotFoundError(
            f"Enhanced image folder not found:\n{image_folder}"
        )

    all_images = sorted([
        name for name in os.listdir(image_folder)
        if name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    if len(all_images) < 10000:
        raise ValueError(
            f"Expected at least 10,000 images; "
            f"found {len(all_images)}."
        )

    print("Available images:", len(all_images))

    # Check existing Model 4 and Model 5 files before training.
    for model_number in DATASET_RANGES:

        output_path = os.path.join(
            RESULTS_FOLDER,
            f"model_epoch_{model_number}.pth"
        )

        if os.path.exists(output_path):
            raise FileExistsError(
                f"Model {model_number} checkpoint already exists:\n"
                f"{output_path}\n"
                "No files were overwritten. Inspect the checkpoint first."
            )

    # Sequentially train Model 4 and Model 5.
    for model_number, (start_index, end_index) in (
        DATASET_RANGES.items()
    ):

        print("\n" + "=" * 60)
        print(f"STARTING MODEL {model_number}")
        print("=" * 60)

        dataset = VOCDataset(
            start_index=start_index,
            end_index=end_index
        )

        if len(dataset) != 2000:
            raise ValueError(
                f"Model {model_number}: expected 2,000 images, "
                f"found {len(dataset)}."
            )

        train_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Create the model architecture.
        model = get_model()

        # Load the preceding model's learned weights.
        print("Loading checkpoint:", previous_checkpoint)

        checkpoint = torch.load(
            previous_checkpoint,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(checkpoint)
        model.to(device)

        # Train the current model.
        train_one_model(
            model,
            train_loader,
            model_number
        )

        # Save the completed checkpoint.
        output_path = os.path.join(
            RESULTS_FOLDER,
            f"model_epoch_{model_number}.pth"
        )

        torch.save(
            model.state_dict(),
            output_path
        )

        print("Saved checkpoint:", output_path)

        # Model 5 starts from Model 4.
        previous_checkpoint = output_path

        del model
        del train_loader
        del dataset

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("MODEL 4 AND MODEL 5 TRAINING FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()