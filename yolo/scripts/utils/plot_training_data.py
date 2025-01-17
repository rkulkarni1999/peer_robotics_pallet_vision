import pandas as pd
import matplotlib.pyplot as plt

# Load results.csv
# results = pd.read_csv("runs/detect/detection_train/results.csv")
results = pd.read_csv("runs/segment/segmentation_train/results.csv")

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

# Plot Training and Validation Losses
axes[0].plot(results["epoch"], results["train/box_loss"], label="Train Box Loss")
axes[0].plot(results["epoch"], results["val/box_loss"], label="Validation Box Loss")
axes[0].plot(results["epoch"], results["train/cls_loss"], label="Train Class Loss")
axes[0].plot(results["epoch"], results["val/cls_loss"], label="Validation Class Loss")
axes[0].set_title("Training and Validation Losses")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid()

# Plot Mean Average Precision (mAP)
axes[1].plot(results["epoch"], results["metrics/mAP50(B)"], label="mAP@50")
axes[1].plot(results["epoch"], results["metrics/mAP50-95(B)"], label="mAP@50-95")
axes[1].set_title("Mean Average Precision (mAP)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("mAP")
axes[1].legend()
axes[1].grid()

# Plot Precision and Recall
axes[2].plot(results["epoch"], results["metrics/precision(B)"], label="Precision", color="blue")
axes[2].plot(results["epoch"], results["metrics/recall(B)"], label="Recall", color="green")
axes[2].set_title("Precision and Recall")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Metrics")
axes[2].legend()
axes[2].grid()

# Adjust layout
plt.tight_layout()

# Show the plots
# plt.show()

# plt.savefig("runs/detect/detection_train/plots/plot_training.jpg")
plt.savefig("runs/segment/segmentation_train/plots/plot_training.jpg")
# Get the final epoch's mAP metrics
final_map_50 = results["metrics/mAP50(B)"].iloc[-1]
final_map_50_95 = results["metrics/mAP50-95(B)"].iloc[-1]

print(f"Final mAP@50: {final_map_50:.3f}")
print(f"Final mAP@50:95: {final_map_50_95:.3f}")