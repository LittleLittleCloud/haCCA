
from .data import Data
import numpy as np
from sklearn.metrics import adjusted_rand_score


def label_transfer_ari(source: Data, target: Data):
    """
    Compute the Adjusted Rand Index (ARI) between the labels of the source and target datasets.
    
    Parameters:
    - source: Data instance, the source dataset with original labels.
    - target: Data instance, the target dataset with transferred labels.
    
    Returns:
    - ARI: The Adjusted Rand Index, measuring the similarity between the source and target labels.
    """
    # Assuming source.Label and target.Label are numpy arrays of labels
    ari_score = adjusted_rand_score(source.Label, target.Label)
    return ari_score

def loss(predict: Data, target: Data, alpha: float = 0.5) -> float:
    """
    Calculate the loss between the predict and target data
    :param predict: Data, the predict data
    :param target: Data, the target data
    :param alpha: float, the balance parameter
    :return: float, the loss
    """
    # calculate the L2 loss between the predict and target D matrix
    loss_D = np.linalg.norm(predict.X - target.X, ord=2)

    # calculate the label match accuracy
    same_values_mask = predict.Label == target.Label
    accuracy = sum(same_values_mask) / len(same_values_mask)
    ari = label_transfer_ari(predict, target)
    
    return loss_D, accuracy, ari