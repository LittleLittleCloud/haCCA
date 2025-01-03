import numpy as np

from hacca.utils import _generate_mock_data
from .loss import pairwise_alignment_accuracy, label_transfer_ari
from .alignment import direct_alignment_metric


def test_pairwise_alignment_accuracy():
    # Create mock data
    labels = ["A", "B", "C"]
    a = _generate_mock_data(20, 3, labels)
    b_prime = _generate_mock_data(30, 2, labels)
    pi = direct_alignment_metric(a, b_prime)

    accuracy = pairwise_alignment_accuracy(a, b_prime, pi)
    print(accuracy)
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"

def test_label_transfer_ari():
    # Create mock data
    labels = ["A", "B", "C"]
    a = _generate_mock_data(20, 3, labels)
    b = _generate_mock_data(20, 3, labels)

    ari = label_transfer_ari(a, b)
    print(ari)
    assert -1 <= ari <= 1, "ARI should be between -1 and 1"