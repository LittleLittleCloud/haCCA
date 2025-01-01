import pytest
import numpy as np

from .alignment import direct_alignment, Data, direct_alignment_with_k_nearest_neighbors, icp_2d_with_feature_alignment

def test_direct_alignment():
    # Create mock data
    labels_a = ["A", "B"]
    labels_b = ["C", "D", "E"]
    a = _generate_mock_data(2, 3, labels_a)
    b_prime = _generate_mock_data(3, 2, labels_b)

    # Call direct_alignment
    aligned_b = direct_alignment(a, b_prime, work_dir=None, enable_center_and_scale=False)

    # Assert aligned_a has the same shape as a but different content if any alignment occurred
    assert aligned_b.X.shape == (3, 3), "Aligned data should have the same shape in X dimension"
    assert aligned_b.D.shape == b_prime.D.shape, "Aligned data should have the same shape in D dimension"
    assert len(aligned_b.Label) == len(labels_b), "Aligned data should have the same number of labels"

    #aligned_a's X's dimension should be the same as X_a's dimension
    print(aligned_b.X.shape)
    assert aligned_b.X.shape[1] == 3, "Aligned data should have the same number of features as a"

    # aligned_a.label should be in one of [A, B]
    assert all([label in labels_a for label in aligned_b.Label]), "Aligned data should have labels in a"


def test_direct_alignment_with_k_nearest_neighbors():
    # Create mock data
    labels_a = ["A", "B"]
    labels_b = ["C", "D", "E"]
    a = _generate_mock_data(2, 3, labels_a)
    b_prime = _generate_mock_data(3, 2, labels_b)

    # Call direct_alignment
    aligned_bs = direct_alignment_with_k_nearest_neighbors(a, b_prime, work_dir=None, enable_center_and_scale=False, k=2)

    # aligned_bs's length should be the same as the number of k
    assert len(aligned_bs) == 2, "Aligned data should have the same number of k"

    for aligned_b in aligned_bs:
        # Assert aligned_a has the same shape as a but different content if any alignment occurred
        assert aligned_b.X.shape == (3, 3), "Aligned data should have the same shape in X dimension"
        assert aligned_b.D.shape == b_prime.D.shape, "Aligned data should have the same shape in D dimension"
        assert len(aligned_b.Label) == len(labels_b), "Aligned data should have the same number of labels"

        #aligned_a's X's dimension should be the same as X_a's dimension
        print(aligned_b.X.shape)
        assert aligned_b.X.shape[1] == 3, "Aligned data should have the same number of features as a"

        # aligned_a.label should be in one of [A, B]
        assert all([label in labels_a for label in aligned_b.Label]), "Aligned data should have labels in a"

def test_icp_2d_with_feature_alignemnt():
    # Create mock data
    labels_a = ["A", "B"]
    labels_b = ["C", "D", "E"]
    a = _generate_mock_data(200, 30, labels_a)
    b_prime = _generate_mock_data(300, 20, labels_b)

    a_, b_prime_ = icp_2d_with_feature_alignment(
        a, b_prime,
        work_dir=None,
        simpson_index_threshold=0.99,
        n_components=1,
    )

    # a_'s X's dimension should be the same as X_a's dimension
    assert a_.X.shape[1] == 30, "Aligned data should have the same number of features as a"

    # a_'s label should be in one of [A, B]
    assert all([label in labels_a for label in a_.Label]), "Aligned data should have labels in a"

    # b_prime_'s X's dimension should be the same as X_b_prime's dimension
    assert b_prime_.X.shape[1] == 20, "Aligned data should have the same number of features as b_prime"

    # b_prime_'s label should be in one of [C, D, E]
    assert all([label in labels_b for label in b_prime_.Label]), "Aligned data should have labels in b_prime"




def _generate_mock_data(n: int, feature_n: int, labels):
    """
    Generate [n, feature_n] data and [n, 2] distance matrix with random selection of labels
    """

    X = np.random.rand(n, feature_n)
    D = np.random.rand(n, 2)
    label_indices = np.random.randint(0, len(labels), n)
    labels = np.array([labels[i] for i in label_indices])

    return Data(X=X, D=D, Label=labels)
