
from .data import Data
import numpy as np
from sklearn.metrics import adjusted_rand_score

def pairwise_alignment_accuracy(source: Data, target: Data, pi: np.ndarray) -> float:
    """
    Calculate the pairwise alignment accuracy between the source and target data.
    :param source: Data, the source data
    :param target: Data, the target data
    :param pi: np.ndarray, the alignment matrix [n, m] where n is the number of data points in the source data and m is the number of data points in the target data.
    :return: float, the pairwise alignment accuracy, would be a value between 0 and 1
    """

    # Create indicator matrix for source and target labels
    # Indicator matrix should be a [n, m] matrix where n is the number of data points in the source data and m is the number of data points in the target data
    # For each pair of data points (i, j), the indicator matrix should have a 1 if the source data point i's label matches the target data point j's label, and 0 otherwise

    indicator_matrix = np.zeros((source.X.shape[0], target.X.shape[0]))
    for i in range(source.X.shape[0]):
        for j in range(target.X.shape[0]):
            indicator_matrix[i, j] = source.Label[i] == target.Label[j]
    
    # Calculate the pairwise alignment accuracy
    accuracy = np.sum(np.sum(pi * indicator_matrix / np.sum(pi, axis=1, keepdims=True))) / source.X.shape[0]

    return accuracy

def label_transfer_ari(source: Data, target: Data):
    """
    Compute the Adjusted Rand Index (ARI) between the labels of the source and target datasets. Closer to 1 means more similar labels. Closer to -1 means more dissimilar labels.
    
    Parameters:
    - source: Data instance, the source dataset with original labels.
    - target: Data instance, the target dataset with transferred labels.
    
    Returns:
    - ARI: The Adjusted Rand Index, measuring the similarity between the source and target labels.
    """
    # Assuming source.Label and target.Label are numpy arrays of labels
    ari_score = adjusted_rand_score(source.Label, target.Label)
    return ari_score

def loss(predict: Data, target: Data) -> float:
    """
    Calculate the loss between the predict and target data
    :param predict: Data, the predict data
    :param target: Data, the target data
    :param alpha: float, the balance parameter
    :return: float, the loss
    """
    # calculate the L2 loss between the predict and target D matrix
    loss_D = np.linalg.norm(predict.X - target.X, ord=2)

    # calculate the alignment accuracy
    same_values_mask = predict.Label == target.Label
    accuracy = sum(same_values_mask) / len(same_values_mask)
    ari = label_transfer_ari(predict, target)

    # calculate the pairwise alignment accuracy
    
    return loss_D, accuracy, ari