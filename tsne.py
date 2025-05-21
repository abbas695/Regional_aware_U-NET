import numpy as np
import matplotlib.pyplot as plt
import openTSNE
import torch
import torch.nn.functional as F

# Paths to input files
files = {
    "et": "paper/tsne_for_paper/activations/base/activation_et_11.npy",
    "tc": "paper/tsne_for_paper/activations/base/activation_tc_7.npy",
    "wt": "paper/tsne_for_paper/activations/base/activation_wt_3.npy",
}

# Placeholder for final embeddings
final_embeddings = {}

# Perform t-SNE for each file
for region, file_path in files.items():
    print(f"Processing {region.upper()}...")
    
    # Load and reshape data
    array = np.load(file_path)
    reshaped_array = array.reshape(array.shape[0], -1).T
    print(f"Reshaped {region.upper()}: {reshaped_array.shape}")

    # Perform t-SNE
    embedding = openTSNE.TSNE(
        perplexity=500,
        initialization="pca",
        metric="cosine",
        n_jobs=32,
        random_state=0,
        verbose=True,
        n_iter=1000,
    ).fit(reshaped_array)
    
    # Store the result for scatter plot and free memory
    final_embeddings[region] = embedding
    del reshaped_array
# Load and downsample the ground truth
ground_truth = np.load('paper/tsne_for_paper/ground truth/ground_truth_1356_0.npy')
ground_truth_tensor = torch.from_numpy(ground_truth).float()

# Assuming all input arrays have the same spatial dimensions
target_shape = array.shape[1:]   # Replace with appropriate dimensions if different
downsampled_tensor = F.interpolate(ground_truth_tensor, size=target_shape).squeeze()
labels = downsampled_tensor.numpy().flatten()

print("Ground truth shape:", labels.shape)
# Plot
plt.figure(figsize=(12, 12))

# Scatter plot for each category
plt.scatter(
    final_embeddings["tc"][labels == 1][:, 0],
    final_embeddings["tc"][labels == 1][:, 1],
    c="blue",
    s=5,
    label="Necrosis (from TC)",
    alpha=0.6,
)
plt.scatter(
    final_embeddings["wt"][labels == 2][:, 0],
    final_embeddings["wt"][labels == 2][:, 1],
    c="green",
    s=5,
    label="Edema (from WT)",
    alpha=0.6,
)
plt.scatter(
    final_embeddings["et"][labels == 3][:, 0],
    final_embeddings["et"][labels == 3][:, 1],
    c="red",
    s=5,
    label="Enhancing Tumor (from ET)",
    alpha=0.6,
)

# Plot formatting
plt.title("base_96")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Tumor Regions", scatterpoints=1, markerscale=5)
plt.show()
