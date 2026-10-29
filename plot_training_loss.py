import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = r"C:\LowLightObstacleDetection"

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "retrained",
    "training_history.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "training_loss.png"
)

if not os.path.exists(CSV_PATH):
    print("Training history not found:", CSV_PATH)
    print("Run a training script that records loss first.")
    raise SystemExit(1)

data = pd.read_csv(CSV_PATH)

data["training_loss"] = pd.to_numeric(
    data["training_loss"],
    errors="coerce"
)

data = data.dropna(subset=["training_loss"])

plt.figure(figsize=(8, 5))

plt.plot(
    data["epoch"],
    data["training_loss"],
    marker="o",
    label="Training Loss"
)

plt.xlabel("Model / Dataset Group")
plt.ylabel("Average Training Loss")
plt.title("Training Loss Across Dataset Groups")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print("Training-loss graph saved to:", OUTPUT_PATH)