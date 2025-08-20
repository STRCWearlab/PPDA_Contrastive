import errno
import os
import random
import numpy as np
import torch

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from matplotlib import colormaps

import wandb


def makedir(path):
    """
    Creates a directory if not already exists.

    :param str path: The path which is to be created.
    :return: None
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if not os.path.exists:
        print(f"[+] Created directory in {path}")


def paint(text, color="green"):
    """
    :param text: string to be formatted
    :param color: color used for formatting the string
    :return:
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    if color == "blue":
        return OKBLUE + text + ENDC
    elif color == "green":
        return OKGREEN + text + ENDC


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def seed_torch(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.use_deterministic_algorithms(True)


def extract_representations(model, data_loader, device):
    """
    Extracts representations (embeddings) from the pre-trained model.

    :param model: The pre-trained backbone (without classifier head).
    :param data_loader: DataLoader for the dataset.
    :param device: Device (cuda or cpu).
    :return: (embeddings, labels)
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, _ in tqdm(data_loader, desc="Extracting representations"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass through the model
            embeddings = model.backbone(inputs)  # Extract latent representations

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_embeddings, all_labels


def sample_equal_per_class(embeddings, labels, num_samples_per_class):
    """
    Samples an equal number of samples from each class for balanced t-SNE visualization.

    :param embeddings: High-dimensional embeddings (numpy array or similar).
    :param labels: Corresponding class labels.
    :param num_samples_per_class: Maximum number of samples to retain per class.
    :return: Balanced embeddings and labels.
    """
    sampled_embeddings = []
    sampled_labels = []

    # Iterate over each unique class
    unique_classes = np.unique(labels)
    for cls in unique_classes:
        # Get indices of all samples belonging to the current class
        class_indices = np.where(labels == cls)[0]

        # Randomly sample `num_samples_per_class` indices
        sampled_indices = np.random.choice(
            class_indices,
            size=min(num_samples_per_class, len(class_indices)),
            replace=False,
        )

        # Append the sampled embeddings and labels
        sampled_embeddings.append(embeddings[sampled_indices])
        sampled_labels.append(labels[sampled_indices])

    # Concatenate results
    sampled_embeddings = np.vstack(sampled_embeddings)
    sampled_labels = np.concatenate(sampled_labels)

    return sampled_embeddings, sampled_labels


def plot_tsne(
    embeddings,
    labels,
    perplexity=30,
    n_iter=1000,
    learning_rate=200,
    title="t-SNE Visualization of Representations",
    random_state=1,
):
    """
    Plots a 2D t-SNE visualization of the learned representations.

    :param embeddings: Extracted embeddings (high-dimensional feature vectors).
    :param labels: Corresponding labels.
    :param num_classes: Number of activity classes.
    :param perplexity: t-SNE perplexity.
    :param n_iter: Number of iterations.
    :param learning_rate: Learning rate.
    :param random_state: Random state for reproducibility.
    :param title: Plot title.
    """
    print(
        f"Running t-SNE with perplexity={perplexity}, n_iter={n_iter}, learning_rate={learning_rate}..."
    )

    # Generate a color map for 34 classes
    num_classes = len(np.unique(labels))
    colors = colormaps["Spectral"]

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
    )
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 9.5))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        cmap=colors,
        c=labels,
        alpha=0.7,
    )
    legend_handles = [
        mpatches.Patch(color=colors(i / num_classes), label=f"Class {i}")
        for i in range(num_classes)
    ]
    plt.legend(
        legend_handles,
        [f"Class {i}" for i in range(num_classes)],
        loc="best",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.0,
    )

    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()
