import numpy as np
import cv2
import cv2.ximgproc
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_mse(pred, gt):
    """
    Compute Mean Squared Error (MSE) using the distance transform.
    :param pred: Predicted binary skeleton (numpy array).
    :param gt: Ground-truth binary skeleton (numpy array).
    :return: MSE value.
    """
    # Ensure binary input (0 or 1)
    pred_binary = (pred > 0.5).astype(np.uint8)
    gt_binary = (gt > 0.5).astype(np.uint8)

    # Compute distance transforms
    pred_dist = distance_transform_edt(1 - pred_binary)  # Distance from background
    gt_dist = distance_transform_edt(1 - gt_binary)      # Distance from background

    # Compute MSE
    mse = np.mean((pred_dist - gt_dist) ** 2) / (pred.shape[0] * pred.shape[1])
    return mse


def get_nodes(binary_image, valence):
    if isinstance(binary_image, torch.Tensor):
        binary_image = binary_image.detach().cpu().numpy()

    if binary_image.ndim == 3 and binary_image.shape[0] == 1:
        binary_image = binary_image.squeeze(0)
    elif binary_image.ndim != 2:
        raise ValueError(f"Expected 2D binary image, got shape {binary_image.shape}")

    # Binarize and convert to uint8
    binary_image = (binary_image > 0.5).astype(np.uint8) * 255

    # Apply thinning
    skeleton = cv2.ximgproc.thinning(binary_image)

    # Ensure binary skeleton for neighbor count
    skeleton_bin = (skeleton > 0).astype(np.uint8)

    # Count 8-connected neighbors
    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(skeleton_bin, -1, kernel) - skeleton_bin

    # DEBUG
    print(
        f"[DEBUG] valence={valence}, skeleton pixels: {np.count_nonzero(skeleton_bin)}, matching nodes: {(neighbor_count == valence).sum()}"
    )

    return np.argwhere(neighbor_count == valence)


def compute_node_precision_recall(pred, gt, valence):
    pred_nodes = get_nodes(pred, valence)
    gt_nodes = get_nodes(gt, valence)

    if len(pred_nodes) == 0 or len(gt_nodes) == 0:
        return 0.0, 0.0

    pred_tree = cKDTree(pred_nodes)
    gt_tree = cKDTree(gt_nodes)

    pred_matches = pred_tree.query_ball_tree(gt_tree, r=2)
    gt_matches = gt_tree.query_ball_tree(pred_tree, r=2)

    precision = np.sum([len(m) > 0 for m in pred_matches]) / len(pred_nodes)
    recall = np.sum([len(m) > 0 for m in gt_matches]) / len(gt_nodes)

    return precision, recall


def evaluate_metrics(loader, model):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_batches = 0

    criterion = nn.BCEWithLogitsLoss()

    node_sums = {
        valence: {"precision": 0.0, "recall": 0.0, "count": 0} for valence in range(5)
    }

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()  # Correct thresholding

            # Compute BCE loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_batches += 1

            # Convert to numpy for metric computation
            preds_np = preds.squeeze(1).cpu().numpy()  # shape: (B, H, W)
            labels_np = labels.squeeze(1).cpu().numpy()

            for pred, gt in zip(preds_np, labels_np):
                total_mse += compute_mse(pred, gt)

                for valence in range(5):
                    precision, recall = compute_node_precision_recall(pred, gt, valence)
                    node_sums[valence]["precision"] += precision
                    node_sums[valence]["recall"] += recall
                    node_sums[valence]["count"] += 1

    avg_loss = total_loss / total_batches
    avg_mse = total_mse / total_batches

    avg_node_metrics = {
        valence: {
            "precision": (
                node_sums[valence]["precision"] / node_sums[valence]["count"]
                if node_sums[valence]["count"] > 0
                else 0.0
            ),
            "recall": (
                node_sums[valence]["recall"] / node_sums[valence]["count"]
                if node_sums[valence]["count"] > 0
                else 0.0
            ),
        }
        for valence in range(5)
    }

    return avg_loss, avg_mse, avg_node_metrics
